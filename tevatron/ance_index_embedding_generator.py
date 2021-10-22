import argparse
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from typing import Optional
from transformers import RobertaTokenizer, PreTrainedModel, RobertaConfig, RobertaModel, BertTokenizer
import torch
# from transformers.file_utils import requires_pytorch


class AnceEncoder(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'
    load_tf_weights = None
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        # requires_pytorch(self)
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


def encode_passage(texts, tokenizer, model, device='cuda:0'):
    max_length = 512  # hardcode for now
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding='longest',
        truncation=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    inputs.to(device)
    embeddings = model(inputs["input_ids"]).detach().cpu().numpy()
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder name or path', required=True)
    parser.add_argument('--dimension', type=int, help='dimension of passage embeddings', required=False, default=768)
    parser.add_argument('--corpus', type=str,
                        help='directory that contains corpus files to be encoded, in jsonl format.', required=True)
    parser.add_argument('--index', type=str, help='directory to store brute force index of corpus', required=True)
    parser.add_argument('--batch', type=int, help='batch size', default=32)
    parser.add_argument('--device', type=str, help='device cpu or cuda [cuda:0, cuda:1...]', default='cuda:0')
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.encoder)
    tokenizer.do_lower_case = True
    model = AnceEncoder.from_pretrained(args.encoder)
    model.to(args.device)

    index = faiss.IndexFlatIP(args.dimension)

    if not os.path.exists(args.index):
        os.mkdir(args.index)

    texts = []
    # with open(os.path.join(args.index, 'docid'), 'w') as id_file:
    for file in sorted(os.listdir(args.corpus)):
        file = os.path.join(args.corpus, file)
        if file.endswith('jsonl'):
            print(f'Loading {file}')
            with open(file, 'r') as corpus:
                for idx, line in enumerate(tqdm(corpus.readlines())):
                    info = json.loads(line)
                    docid = info['id']
                    text = info['contents'].strip().replace('\n', ' ')
                    # id_file.write(f'{docid}\n')
                    texts.append(text.lower())
    for idx in tqdm(range(0, len(texts), args.batch)):
        text_batch = texts[idx: idx+args.batch]
        with torch.no_grad():
            embeddings = encode_passage(text_batch, tokenizer, model, args.device)
        index.add(np.array(embeddings))
    faiss.write_index(index, os.path.join(args.index, 'index'))
