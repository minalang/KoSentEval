# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import logging
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 모델 설정
    model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)  # or 'BM-K/KoSimCSE-roberta-multitask'
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")  # or 'BM-K/KoSimCSE-roberta-multitask'
    embeddings = []
    for sent in batch:
        # 텍스트 인코딩
        #print(sent)
        inputs_output = tokenizer(sent, max_length=64, padding='max_length', truncation=True, return_tensors='pt').to(device)
        embeddings_output = model(**inputs_output, return_dict = False)
        average_embedding = torch.mean(embeddings_output[0], dim=1)
        sentvec = average_embedding.tolist()[0]
        embeddings.append(sentvec)
    embeddings = np.vstack(embeddings)
    #print(embeddings)
    return embeddings

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 1, 'optim': 'adam', 'batch_size': 128,
                                 'tenacity': 5, 'epoch_size': 5}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['Length', 'WordContent' , 'SubjOmission', 'TopConstituents', 'SentType', 'Tense', 'Honorifics', 'Sentiment', 'KorSTS']
    results = se.eval(transfer_tasks)
    print(results)
