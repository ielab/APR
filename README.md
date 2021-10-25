# APR
The repo for the paper [Improving Query Representations for DenseRetrieval with Pseudo Relevance Feedback:A Reproducibility Study]().

## Environment setup
To reproduce the results in the paper, we rely on two open-source IR toolkits: [Pyserini](https://github.com/castorini/pyserini) and [tevatron](https://github.com/texttron/tevatron).

We cloned, merged, and modified the two toolkits in this repo and will use them to train and inference the PRF models. We refer to the original github repos to setup the environment:

Install Pyserini: [https://github.com/castorini/pyserini/blob/master/docs/installation.md](https://github.com/castorini/pyserini/blob/master/docs/installation.md).

Install tevatron: [https://github.com/texttron/tevatron#installation](https://github.com/texttron/tevatron#installation).

You also need MS MARCO passage ranking dataset, including the collection and queries. We refer to the official github [repo](https://github.com/microsoft/MSMARCO-Passage-Ranking) for downloading the data.

## To reproduce ANCE-PRF inference results with the original model checkpoint

The code, dataset, and model for reproducing the ANCE-PRF results presented in the original paper:
>HongChien Yu, Chenyan Xiong, Jamie Callan. [Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback](https://arxiv.org/abs/2108.13454)

have been merged into Pyserini source. Simply just need to follow this [instruction](https://github.com/castorini/pyserini/blob/master/docs/experiments-ance-prf.md), which includes the instructions of downloading the dataset, model checkpoint (provided by the original authors), dense index, and PRF inference.

## To train dense retriever PRF models
We use tevatron to train the dense retriever PRF query encodes that we investigated in the paper.

First, you need to have train queries run files
to build hard negative training set for each DR.

You can use Pyserini to generate run files for [ANCE](https://github.com/castorini/pyserini/blob/master/docs/experiments-ance.md), [TCT-ColBERTv2](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert-v2.md) and [DistilBERT KD TASB](https://github.com/castorini/pyserini/blob/master/docs/experiments-distilbert_tasb.md) by changing the query set flag `--topics` to `queries.train.tsv`.

Once you have the run file, cd to `/tevatron` and run:

```
python make_train_from_ranking.py \
	--ranking_file /path/to/train/run \
	--model_type (ANCE or TCT or DistilBERT) \
	--output /path/to/save/hard/negative
```

Apart from the hard negative training set, you also need the original DR query encoder model checkpoints to initial the model weights. You can download them from Huggingface modelhub: [ance](https://huggingface.co/castorini/tct_colbert-v2-hnp-msmarco), [tct_colbert-v2-hnp-msmarco](https://huggingface.co/castorini/tct_colbert-v2-hnp-msmarco), [distilbert-dot-tas_b-b256-msmarco](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco). Please use the same name as the link in Huggingface modelhub for each of the folders that contain the model.


After you generated the hard negative training set and downloaded all the models, you can kick off the training for DR-PRF query encoders by:

```
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    -m tevatron.driver.train \
    --output_dir /path/to/save/mdoel/checkpoints \
    --model_name_or_path /path/to/model/folder \
    --do_train \
    --save_steps 5000 \
    --train_dir /path/to/hard/negative \
    --fp16 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-6 \
    --num_train_epochs 10 \
    --train_n_passages 21 \
    --q_max_len 512 \
    --dataloader_num_workers 10 \
    --warmup_steps 5000 \
    --add_pooler
```

## To inference dense retriever PRF models


