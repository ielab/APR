#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
import os

import faiss
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def encode_passage(text, tokenizer, model, device='cuda:0'):
    max_length = 154  # hardcode for now
    inputs = tokenizer(
        '[CLS] [D] ' + text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors='pt'
    )
    inputs.to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.detach().cpu().numpy()
    return np.mean(embeddings[:, 4:, :], axis=-2).flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder name or path', required=True)
    parser.add_argument('--dimension', type=int, help='dimension of passage embeddings', required=False, default=768)
    parser.add_argument('--corpus', type=str,
                        help='directory that contains corpus files to be encoded, in jsonl format.', required=True)
    parser.add_argument('--index', type=str, help='directory to store brute force index of corpus', required=True)
    parser.add_argument('--device', type=str, help='device cpu or cuda [cuda:0, cuda:1...]', default='cuda:0')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.encoder)
    model = BertModel.from_pretrained(args.encoder)
    model.to(args.device)

    index = faiss.IndexFlatIP(args.dimension)

    if not os.path.exists(args.index):
        os.mkdir(args.index)

    with open(os.path.join(args.index, 'docid'), 'w') as id_file:
        for file in sorted(os.listdir(args.corpus)):
            file = os.path.join(args.corpus, file)
            if file.endswith('json'):
                print(f'Encoding {file}')
                with open(file, 'r') as corpus:
                    for idx, line in enumerate(tqdm(corpus.readlines())):
                        info = json.loads(line)
                        docid = info['id']
                        text = info['contents']
                        id_file.write(f'{docid}\n')
                        embedding = encode_passage(text, tokenizer, model, args.device)
                        index.add(np.array([embedding]))
    faiss.write_index(index, os.path.join(args.index, 'index'))