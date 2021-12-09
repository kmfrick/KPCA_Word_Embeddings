import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

import sys
import io

from kernel_pca import get_huggingface_dataset, get_vocabulary, thresh_vocab
import matplotlib.pyplot as plt
from datasets import load_dataset
import subprocess
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import torch

from torchw2vec.word2vec.model import SkipGramModel
from torchw2vec.word2vec.data_reader import DataReader, Word2vecDataset

from tokenizers import Regex
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

from collections import Counter
import nltk
import numpy as np

from itertools import chain

# Progress bar
from tqdm import tqdm

PATH_TO_SENTEVAL = '../SentEval/'
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# ------------------------------------------------------------------------------------
# FUNCTIONS

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    return word_vec

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

def prepare(params, samples, word2id, fname):
    params.word2id = word2id
    params.word_vec = get_wordvec(fname, params.word2id)
    params.wvec_dim = 128
    return


def main():
# ------------------------------------------------------------------------------------
# SCRIPT
#
    word2id = dict()
    id2word = dict()
    files = os.listdir('.')
    for i, filename in enumerate(files):
        print(f'({i+1}) {filename}')
    fname = files[int(input('Select a file from the list above: ')) - 1]
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
    reduce_with_kpca = False
    if reduce_with_kpca:
        print('Building embedding matrix from training vocabulary...')
        get_caseins_key = lambda x: [i for i in word2id.keys() if i.lower() == x][0]
        S = np.array([emb[i, :] for i in tqdm([word2id[get_caseins_key(w)] for w in training_vocab])])
        gamma_range_center = 1/(S.shape[0])
    else:
        gamma_range_center = 1 / 944
    gridsearch = False
    if gridsearch:
        for gamma_i in range(-3, 3):
            gamma = gamma_range_center + gamma_i * 1e-3
            for n_components in [8, 16, 32, 64]:
                m = S.shape[0]
                avg_acc = 0
                # Transform the similarity matrix using kernel PCA
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
        exit()
    if reduce_with_kpca:
        n_components = 16
        gamma = gamma_range_center - 1e-3
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

# Set params for SentEval
    params_senteval = {'task_path': "../SentEval/data", 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                     'tenacity': 3, 'epoch_size': 2}

# benchmark for semantic evaluation
    def prep(params, samples):
        return prepare(params, samples, word2id, fname)
    se = senteval.engine.SE(params_senteval, batcher, prep)
    transfer_tasks = ['STS12']
# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = se.eval(transfer_tasks)
    print(results["STS12"]["all"])

# Run 5-NN on K
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(emb)
# Get neighbors of a test word
    for test_word in ['history', 'Moscow', 'linguistically', 'statistics', 'firecracker']:
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
        print(f's({test_adverb}, {test_adjective}) = {dist}')

    exit()
# Use TSNE to plot 5-NN in 2D
    tsne = TSNE(
        n_components=2,
        random_state=50321,
    )
    print('Fitting TSNE...')
    pos = tsne.fit_transform(emb)
    print(pos.shape)

    fig = plt.figure(1)
    plt.scatter(pos[neigh_list, 0], pos[neigh_list, 1], color='turquoise')
    plt.scatter(pos[test_id, 0], pos[test_id, 1], color='red')

    for j, xy in zip(neigh_list, pos[neigh_list, :]):
        plt.annotate(id2word[j], xy = xy, textcoords='data')

    plt.title(f'Neighbors of {test_word}')

    plt.show()

if __name__ == '__main__':
    main()
