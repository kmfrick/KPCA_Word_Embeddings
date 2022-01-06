#!/usr/bin/env python

# KERNEL PCA FOR WORD EMBEDDINGS
# Copyright (C) Kevin Michael Frick, Sonia Petrini, Patrick Lutz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from datasets import load_dataset
from kernel_pca import get_huggingface_dataset, get_vocabulary, mse_scorer
from scipy.spatial import distance
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from tokenizers import Regex, Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm
import io
import nltk
import numpy as np
import os
import subprocess
import sys
import torch

PATH_TO_SENTEVAL = './SentEval/'
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# ------------------------------------------------------------------------------------
# FUNCTIONS


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

def prepare(params, samples, emb, dim, word2id, fname):
    params.word2id = word2id

    params.word_vec = dict()
    for w, id in word2id.items():
        params.word_vec[w] = emb[id, :]
    params.wvec_dim = dim
    return


def main():
# ------------------------------------------------------------------------------------
# SCRIPT
#
    huggingface_dataset = get_huggingface_dataset()
    # tokenize the huggingface_dataset -> single words from sentences
    VOCAB_SIZE = 1000
    tokenizer = get_vocabulary(huggingface_dataset['train'], vocab_size = VOCAB_SIZE)
    training_vocab = tokenizer.get_vocab()
    word2id = dict()
    id2word = dict()
    files = os.listdir('.')
    for i, filename in enumerate(files):
        print(f'({i+1}) {filename}')
    file_ids = input('List the file names you want to test from the list above: ').split()
    file_ids = [int(f) - 1 for f in file_ids]
    file_names = [files[i] for i in file_ids]
    print(file_names)
    for fname in file_names:
        print(f'Opening {fname}')
        i = -1
        with open(fname) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                # Skip first line
                if i < 0:
                    print(line)
                    m = int(line.split()[0])
                    p = int(line.split()[1])
                    emb = np.zeros([m, p])
                else:
                    linesp = line.split()
                    word = linesp[0]
                    emb[i] = np.array([float(w) for w in linesp[1:]])
                    word2id[word] = i
                    id2word[i] = word
                i += 1
        # Change this line to perform dimensionality reduction
        reduce_with_kpca = False
        if reduce_with_kpca:
            print('Building embedding matrix for words in training vocabulary...')
            # exclude one-letter words
            S = np.array([emb[i, :] for i in tqdm([word2id[w] for w in training_vocab if len(w) > 1])])
            gamma_c = 1/(S.shape[0])
        gridsearch = False
        if gridsearch and reduce_with_kpca:
            param_grid = [{
                'gamma': np.linspace(1e-6, 1e-2, 20),
                'n_components': [8, 16, 32, 64]
            }]

            kpca = KernelPCA(fit_inverse_transform = True, n_jobs = -1)
            clf = GridSearchCV(kpca, param_grid, cv = 5, scoring = mse_scorer)
            clf.fit(S)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

            exit()
            # Cosine reconstruction error
            for gamma in np.linspace(1e-6, 1e-2, 20):
                for n_components in [8, 16, 32, 64]:
                    m = S.shape[0]
                    avg_acc = 0
                    try:
                        kpca = KernelPCA(kernel='rbf', n_components = n_components, gamma = gamma)
                        K = kpca.fit_transform(S)
                        for i in tqdm(range(m)):
                            for j in range(m):
                                if i != j:
                                    si = S[i, :]
                                    sj = S[j, :]
                                    cos_dist_s = distance.cosine(si, sj)
                                    ki = K[i, :]
                                    kj = K[j, :]
                                    cos_dist_kernel = distance.cosine(ki, kj)
                                    rel_err = (cos_dist_kernel - cos_dist_s) / cos_dist_s
                                    avg_acc += np.abs(rel_err)
                        print(f'gamma = {gamma}; n_comp = {n_components}, avg rel err = {avg_acc / (m ** 2)}')
                    except (ValueError, np.linalg.LinAlgError) as e:
                        print(f'gamma = {gamma}; n_comp = {n_components}; Unfeasible PCA')
                        continue
            # NRMSE
        if reduce_with_kpca:
            # Change this line to change the number of kPCs to retain
            n_components = 64
            gamma = gamma_c
            print(f'Performing KPCA for dimensionality reduction to {n_components}...')
            kpca = KernelPCA(kernel='rbf', n_components = n_components, gamma = gamma)
            kpca.fit(S)
            print(f'Fitting on matrix S, shape = {S.shape}')
            print(f'Transforming a matrix E, shape = {emb.shape}')
            K_new = kpca.transform(emb)
            print(f'Transformed a matrix K, shape = {K_new.shape}')
            # Sanity check: print two embeddings that should be different
            print(K_new[10, :])
            print(K_new[280, :])
            emb = K_new
        else:
            n_components = 128

# Set params for SentEval
        params_senteval = {'task_path': './SentEval/data', 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

# benchmark for semantic evaluation
        def prep(params, samples):
            return prepare(params, samples, emb, n_components, word2id, fname)
        se = senteval.engine.SE(params_senteval, batcher, prep)
        transfer_tasks = ['STS12']
        results = se.eval(transfer_tasks)
        print(fname)
        scc = results['STS12']['all']['spearman']['mean']
        print(f'Spearman correlation coefficient: {scc}')

# Run 5-NN on K
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(emb)
# Get neighbors of a test word
        for test_word in ['history', 'moscow', 'linguistically', 'statistics', 'firecracker']:
            print(f'test_word = {test_word}')
            test_id = word2id[test_word]
            test_embedding = emb[test_id]
            neigh_dist, neigh = nbrs.kneighbors(test_embedding.reshape(1, -1), k+1)
            neigh_list = np.squeeze(neigh)[1:]
            print(neigh_list)
            print([id2word[i] for i in neigh_list])

        adjectives = ['quick', 'statistical', 'calm', 'thorough']
        adverbs = ['quickly', 'statistically', 'calmly', 'thoroughly']
        for test_adverb, test_adjective in zip(adverbs, adjectives):
            emb_adv = emb[word2id[test_adverb]]
            emb_adj = emb[word2id[test_adjective]]
            dist = distance.cosine(emb_adv, emb_adj)
            print(f'd({test_adverb}, {test_adjective}) = {dist}')

if __name__ == '__main__':
    main()
