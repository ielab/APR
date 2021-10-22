import logging
import os
import sys

import datasets
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    get_linear_schedule_with_warmup
)
import torch
from torch.optim import Optimizer

from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data import TrainDataset, QPCollator
from tevatron.preprocessor import HFTrainPreProcessor
from tevatron.modeling import DenseModel
from tevatron.trainer import DenseTrainer as Trainer, GCTrainer

logger = logging.getLogger(__name__)


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    if 'ance' in model_args.model_name_or_path:
        tokenizer.do_lower_case = True
    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.train_dir is not None:
        train_dataset = TrainDataset(
            data_args, model_args, data_args.train_path, tokenizer
        )
    else:
        train_dataset = datasets.load_dataset(data_args.dataset_name)[data_args.dataset_split]
        train_dataset = train_dataset.map(
            HFTrainPreProcessor(tokenizer, data_args.q_max_len, data_args.p_max_len),
            batched=False,
            num_proc=data_args.dataset_proc_num,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        train_dataset = TrainDataset(data_args, model_args, train_dataset, tokenizer)

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer

    if 'ance' in model_args.model_name_or_path:
        optimizer_grouped_parameters = list()
        layer_optim_params = set()
        for layer_name in [
            "roberta.embeddings",
            "score_out",
            "downsample1",
            "downsample2",
            "downsample3"]:
            layer = getattr_recursive(model, layer_name)
            if layer is not None:
                optimizer_grouped_parameters.append({"params": layer.parameters()})
                for p in layer.parameters():
                    layer_optim_params.add(p)
        if getattr_recursive(model, "roberta.encoder.layer") is not None:
            for layer in model.roberta.encoder.layer:
                optimizer_grouped_parameters.append({"params": layer.parameters()})
                for p in layer.parameters():
                    layer_optim_params.add(p)

        optimizer_grouped_parameters.append(
            {"params": [p for p in model.parameters() if p not in layer_optim_params]})

        optimizer = Lamb(optimizer_grouped_parameters, lr=training_args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5000,
            num_training_steps=83240)

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, scheduler),
            data_collator=QPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
        )
    else:
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=QPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
        )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
