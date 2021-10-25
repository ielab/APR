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

Install Pyserini by following the instructions within `pyserini/README.md`

Then run:

```
python -m pyserini.dsearch --topics /path/to/query/tsv/file \
    --index /path/to/index \
    --encoder /path/to/encoder \ # This encoder is for first round retrieval
    --batch-size 64 \
    --output /path/to/output/run/file \
    --prf-method tctv2-prf \
    --threads 12 \
    --sparse-index msmarco-passage \
    --prf-encoder /path/to/encoder \ # This encoder is for PRF query generation
    --prf-depth 3
```

An example would be:
```
python -m pyserini.dsearch --topics ./data/msmarco-test2020-queries.tsv \
    --index ./dindex-msmarco-passage-tct_colbert-v2-hnp-bf \
    --encoder ./tct_colbert_v2_hnp \
    --batch-size 64 \
    --output ./runs/tctv2-prf3.res \
    --prf-method tctv2-prf \
    --threads 12 \
    --sparse-index msmarco-passage \
    --prf-encoder ./tct-colbert-v2-prf3/checkpoint-10000 \
    --prf-depth 3
```

Or one can use pre-built index and models available in Pyserini:

```
python -m pyserini.dsearch --topics dl19-passage \
    --index msmarco-passage-tct_colbert-v2-hnp-bf \
    --encoder castorini/tct_colbert-v2-hnp-msmarco \
    --batch-size 64 \
    --output ./runs/tctv2-prf3.res \
    --prf-method tctv2-prf \
    --threads 12 \
    --sparse-index msmarco-passage \
    --prf-encoder ./tct-colbert-v2-prf3/checkpoint-10000 \
    --prf-depth 3
```

The PRF depth `--prf-depth 3` depends on the PRF encoder trained, if trained with PRF 3, here only can use PRF 3.

Where `--topics` can be:
TREC DL 2019 Passage: `dl19-passage`
TREC DL 2020 Passage: `dl20`
MS MARCO Passage V1: `msmarco-passage-dev-subset`

`--encoder` can be:
ANCE: `castorini/ance-msmarco-passage`
TCT-ColBERT V2 HN+: `castorini/tct_colbert-v2-hnp-msmarco`
DistilBERT Balanced: `sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco`

`--index` can be:
ANCE index with MS MARCO V1 passage collection: `msmarco-passage-ance-bf`
TCT-ColBERT V2 HN+ index with MS MARCO V1 passage collection: `msmarco-passage-tct_colbert-v2-hnp-bf`
DistillBERT Balanced index with MS MARCO V1 passage collection: `msmarco-passage-distilbert-dot-tas_b-b256-bf`

To evaluate the run:

TREC DL 2019
```
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m recall.1000 -l 2 dl19-passage ./runs/tctv2-prf3.res
```

TREC DL 2020
```
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m recall.1000 -l 2 dl20-passage ./runs/tctv2-prf3.res
```

MS MARCO Passage Ranking V1
```
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset ./runs/tctv2-prf3.res
```