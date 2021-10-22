import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist
import sklearn
import numpy as np

from transformers import AutoModel, PreTrainedModel, RobertaConfig, RobertaModel, AutoTokenizer, DistilBertConfig, \
    BertModel, BertTokenizerFast, BertTokenizer, BertConfig
from transformers.modeling_outputs import ModelOutput

if torch.cuda.is_available():
    from torch.cuda.amp import autocast

from typing import Optional, Dict

from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


class TctColBertDocumentEncoder(PreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config)
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def init_weights(self):
        pass

    def encode(self, texts, fp16=False):
        # texts = ['[CLS] [D] ' + text for text in texts]
        # max_length = 512  # hardcode for now
        # inputs = self.tokenizer(
        #     texts,
        #     max_length=max_length,
        #     padding="longest",
        #     truncation=True,
        #     add_special_tokens=False,
        #     return_tensors='pt'
        # )
        if fp16:
            with autocast():
                with torch.no_grad():
                    outputs = self.model(**texts)
        else:
            outputs = self.model(**texts)
        embeddings = self._mean_pooling(outputs["last_hidden_state"][:, 4:, :], texts['attention_mask'][:, 4:])
        return embeddings


class TctColBertQueryEncoder(PreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config)
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def init_weights(self):
        pass

    def encode(self, query):
        # max_length = 512  # hardcode for now
        # inputs = self.tokenizer(
        #     '[CLS] [Q] ' + query + '[MASK]' * max_length,
        #     max_length=max_length,
        #     truncation=True,
        #     add_special_tokens=False,
        #     return_tensors='pt'
        # )
        outputs = self.model(**query)
        embeddings = outputs.last_hidden_state
        return torch.mean(embeddings[:, 4:, :], dim=-2)


class AnceEncoder(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'
    load_tf_weights = None
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.embeddingHead = torch.nn.Linear(config.hidden_size, 768)
        self.norm = torch.nn.LayerNorm(768)
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.roberta.init_weights()
        self.embeddingHead.apply(self._init_weights)
        self.norm.apply(self._init_weights)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.roberta.config.pad_token_id)
            )
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.norm(self.embeddingHead(pooled_output))
        return pooled_output


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        # self.linear_q = nn.Linear(input_dim, output_dim)
        self.embeddingHead = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(768)
        if tied:
            # self.linear_p = self.linear_q
            pass
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            # return self.linear_q(q[:, 0])
            return self.norm(self.embeddingHead(q[:, 0]))
        elif p is not None:
            # return self.linear_p(p[:, 0])
            return self.norm(self.embeddingHead(p[:, 0]))
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class AutoQueryEncoder(PreTrainedModel):

    def __init__(self, config, encoder_dir):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(encoder_dir)
        # self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
        self.pooling = 'cls'

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def init_weights(self):
        pass

    def encode(self, query):
        # inputs = self.tokenizer.encode_plus(
        #     query,
        #     padding='longest',
        #     truncation=True,
        #     add_special_tokens=True,
        #     return_tensors='pt'
        # )
        # inputs.to(self.device)
        outputs = self.model(**query)
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs, query['attention_mask'])
        else:
            embeddings = outputs[0][:, 0, :]
        return embeddings


class AutoDocumentEncoder(PreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(model_name).eval()
        # self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = 'cls'
        self.l2_norm = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def init_weights(self):
        pass

    def encode(self, texts):
        # inputs = self.tokenizer.encode_plus(
        #     texts,
        #     max_length=512,
        #     padding='longest',
        #     truncation=True,
        #     add_special_tokens=True,
        #     return_tensors='pt'
        # )
        # print(inputs)
        # texts.to(self.device)
        outputs = self.model(**texts)
        if self.pooling == "mean":
            embeddings = self.mean_pooling(outputs[0], texts['attention_mask'])
        else:
            embeddings = outputs[0][:, 0, :]
        if self.l2_norm:
            sklearn.preprocessing.normalize(embeddings, axis=1)
        return embeddings


class DenseModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()
        if 'ance' in model_args.model_name_or_path:
            self.lm_q = AnceEncoder(BertConfig.from_pretrained(
                model_args.model_name_or_path), model_name=model_args.model_name_or_path)
            self.passage_encoder = TctColBertDocumentEncoder(BertConfig.from_pretrained(
                model_args.model_name_or_path), model_name=model_args.model_name_or_path)
        elif 'tct_colbert' in model_args.model_name_or_path:
            self.lm_q = TctColBertQueryEncoder(BertConfig.from_pretrained(
                model_args.model_name_or_path), model_name=model_args.model_name_or_path)
            self.passage_encoder = TctColBertDocumentEncoder(BertConfig.from_pretrained(
                model_args.model_name_or_path), model_name=model_args.model_name_or_path)
        elif 'distilbert' in model_args.model_name_or_path:
            self.lm_q = AutoQueryEncoder(DistilBertConfig.from_pretrained(
                model_args.model_name_or_path), encoder_dir=model_args.model_name_or_path)
            self.passage_encoder = AutoDocumentEncoder(DistilBertConfig.from_pretrained(
            model_args.model_name_or_path), model_name=model_args.model_name_or_path)

        self.lm_p = self.lm_q
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        self.passage_encoder.eval()
        with torch.no_grad():
            p_reps = self.passage_encoder.encode(passage)
        # p_hidden, p_reps = self.encode_passage(passage)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.train_args.negatives_x_device:
                q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
            # scores = scores.view(effective_bsz, -1)

            temp = torch.zeros(self.train_args.per_device_train_batch_size, self.data_args.train_n_passages,
                               device=scores.device)
            for i, line in enumerate(scores):
                temp[i][:] = line[
                             i * self.data_args.train_n_passages:i * self.data_args.train_n_passages + self.data_args.train_n_passages]
            scores = temp

            # target = torch.arange(
            #     scores.size(0),
            #     device=scores.device,
            #     dtype=torch.long
            # )
            # target = target * self.data_args.train_n_passages
            # loss = self.cross_entropy(scores, target)

            loss = self.cross_entropy(scores, self.target_label)

            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

        else:
            loss = None
            if query and passage:
                scores = (q_reps * p_reps).sum(1)
            else:
                scores = None

            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

    # def forward(
    #         self,
    #         query: Dict[str, Tensor] = None,
    #         passage: Dict[str, Tensor] = None,
    # ):
    #
    #     q_hidden, q_reps = self.encode_query(query)
    #     self.ance_encoder.eval()
    #     with torch.no_grad():
    #         p_reps = self.ance_encoder(passage["input_ids"]).detach()
    #     # p_hidden, p_reps = self.encode_passage(passage)
    #
    #     if q_reps is None or p_reps is None:
    #         return DenseOutput(
    #             q_reps=q_reps,
    #             p_reps=p_reps
    #         )
    #
    #     if self.training:
    #         if self.train_args.negatives_x_device:
    #             q_reps = self.dist_gather_tensor(q_reps)
    #             p_reps = self.dist_gather_tensor(p_reps)
    #
    #         effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
    #             if self.train_args.negatives_x_device \
    #             else self.train_args.per_device_train_batch_size
    #
    #         scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
    #         scores = scores.view(effective_bsz, -1)
    #
    #         target = torch.arange(
    #             scores.size(0),
    #             device=scores.device,
    #             dtype=torch.long
    #         )
    #         target = target * self.data_args.train_n_passages
    #         loss = self.cross_entropy(scores, target)
    #         if self.train_args.negatives_x_device:
    #             loss = loss * self.world_size  # counter average weight reduction
    #         return DenseOutput(
    #             loss=loss,
    #             scores=scores,
    #             q_reps=q_reps,
    #             p_reps=p_reps
    #         )
    #
    #     else:
    #         loss = None
    #         if query and passage:
    #             scores = (q_reps * p_reps).sum(1)
    #         else:
    #             scores = None
    #
    #         return DenseOutput(
    #             loss=loss,
    #             scores=scores,
    #             q_reps=q_reps,
    #             p_reps=p_reps
    #         )

    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q.encode(qry)
        # q_hidden = qry_out.last_hidden_state
        # if self.pooler is not None:
        #     q_reps = self.pooler(q=q_hidden)
        # else:
        #     q_reps = q_hidden[:, 0]
        return None, qry_out

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.model.save_pretrained(output_dir)
            self.lm_q.tokenizer.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(DenseModel):
    POOLER_CLS = LinearPooler

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DenseModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DenseModelForInference, self).encode_query(qry)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER_CLS(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler
        )
        return model
