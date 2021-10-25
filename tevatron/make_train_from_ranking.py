import argparse
from tqdm import tqdm
import random
from transformers import RobertaTokenizer, BertTokenizer, DistilBertTokenizer
import json

random.seed(99)


def read_query(query_path):
    query_collection = dict()
    total = sum(1 for _ in open(query_path))
    for query_line in tqdm(open(query_path, 'r'), desc='Load Query', total=total):
        [qid, query] = query_line.strip().split('\t')
        query_collection[qid] = query
    return query_collection


def read_collection(collection_path):
    collection = dict()
    total = sum(1 for _ in open(collection_path))
    for collection_line in tqdm(open(collection_path, 'r'), desc='Load Collection', total=total):
        [pid, passage] = collection_line.strip().split('\t')
        collection[pid] = passage
    return collection


def read_ranking(ranking_path, pair, k, from_top):
    ranking = dict()
    topk = dict()
    total = sum(1 for _ in open(ranking_path))
    for ranking_line in tqdm(open(ranking_path, 'r'), desc='Load Ranking', total=total):
        [qid, _, pid, rank, _, _] = ranking_line.strip().split()
        targets = pair[qid].keys()
        if qid not in ranking:
            if pid not in targets and int(rank) <= from_top:
                ranking[qid] = [pid]
            else:
                ranking[qid] = []
        else:
            if pid not in targets and int(rank) <= from_top:
                ranking[qid].append(pid)
        if int(rank) <= k:
            if qid not in topk:
                topk[qid] = [pid]
            else:
                topk[qid].append(pid)
        else:
            continue
    return ranking, topk


def read_query_target_pair(pair_path):
    pair = dict()
    total = sum(1 for _ in open(pair_path))
    for pair_line in tqdm(open(pair_path, 'r'), desc='Load Q-D Pair', total=total):
        [qid, _, pid, target] = pair_line.strip().split('\t')
        if qid not in pair:
            pair[qid] = {pid: target}
        else:
            pair[qid][pid] = target
    return pair


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranking-file', type=str)
    parser.add_argument('--query-file', type=str, default='./data/msmarco_passage/query/train_query_judged.tsv')
    parser.add_argument('--collection-file', type=str, default='./data/msmarco_passage/collection/collection.tsv')
    parser.add_argument('--pair-file', type=str, default='./data/msmarco_passage/qrels/train_query_passage_pair.tsv')
    parser.add_argument('--encoder', type=str, default='./data/msmarco_passage/models/ANCE')
    parser.add_argument('--model-type', type=str, default='ANCE', help='Can be ANCE, TCT, DistilBERT')
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--from-top', type=int, default=1000)
    parser.add_argument('--prf-k', type=int, default=3)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    queries = read_query(args.query_file)
    collection = read_collection(args.collection_file)
    pair = read_query_target_pair(args.pair_file)

    if args.model_type == 'ANCE':
        tokenizer = RobertaTokenizer.from_pretrained(args.encoder)
    elif args.model_type == 'DistilBERT':
        tokenizer = DistilBertTokenizer.from_pretrained(args.encoder)
    elif args.model_type == 'TCT':
        tokenizer = BertTokenizer.from_pretrained(args.encoder)
    else:
        raise NotImplementedError()

    rankings, topk = read_ranking(args.ranking_file, pair, args.prf_k, args.from_top)

    fout = open(f'{args.output.rsplit(".", 1)[0]}_from_{args.from_top}_prf{args.prf_k}.{args.output.rsplit(".", 1)[1]}', 'a+')

    for qid in tqdm(pair.keys(), desc='Building Train'):
        original_query = queries[qid]
        if args.prf_k != 0:
            prf_passages = list()
            for pid in topk[qid]:
                prf_passages.append(collection[pid])

            prf_passages_concatenation = f'{tokenizer.sep_token}'.join(prf_passages)
            if args.model_type == 'TCT':
                prf_query = f'{tokenizer.cls_token} [Q] {original_query}{tokenizer.sep_token}{prf_passages_concatenation}{tokenizer.mask_token * 512}'
            else:
                prf_query = f'{original_query}{tokenizer.sep_token}{prf_passages_concatenation}'
        else:
            prf_query = original_query

        for pid in pair[qid].keys():
            positive_passage = collection[pid]
            if args.sample > len(rankings[qid]):
                sample = len(rankings[qid])
            else:
                sample = args.sample
            random_negatives = random.sample(rankings[qid], k=sample)
            if args.model_type == 'TCT':
                negatives = [f'{tokenizer.cls_token} [D] {collection[d]}' for d in random_negatives]
                train_instance = {
                    'query': prf_query,
                    'positives': [f'{tokenizer.cls_token} [D] {positive_passage}'],
                    'negatives': negatives
                }
            else:
                negatives = [collection[d] for d in random_negatives]
                train_instance = {
                    'query': prf_query,
                    'positives': [positive_passage],
                    'negatives': negatives
                }
            fout.write(f'{json.dumps(train_instance)}\n')
