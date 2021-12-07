import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

import sys
import io

PATH_TO_SENTEVAL = './SentEval/'
PATH_TO_VEC = 'Embeddings_128_InjectTrue_text_4Epochs.vec'
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

word2id = dict()
id2word = dict()

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

def prepare(params, samples, word2id=word2id):
    params.word2id = word2id
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 128
    return


def cosine_similarity(vec1, vec2):
    """
    given 2 np.arrays -> calculate the cosine similarity value
    """
    similarity = vec1.dot(vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity
# ------------------------------------------------------------------------------------
# SCRIPT
#
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

# Set params for SentEval
params_senteval = {'task_path': "./SentEval/data", 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# benchmark for semantic evaluation
se = senteval.engine.SE(params_senteval, batcher, prepare)
transfer_tasks = ['STS12']
# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
results = se.eval(transfer_tasks)
print(results)

# Get neighbors of a test word
test_word = input('Input a test word: ')
print(f'test_word = {test_word}')
test_id = word2id[test_word]
print(f"the id of the input word is {test_id}")
test_embedding = emb[test_id]
# Run 5-NN on K
k = 5
nbrs = NearestNeighbors(n_neighbors=k)
nbrs.fit(emb)
neigh_dist, neigh = nbrs.kneighbors(test_embedding.reshape(1, -1), k+1)
neigh_list = np.squeeze(neigh)[1:]
print(neigh_dist)
print(neigh_list)
print([id2word[i] for i in neigh_list])

test_adverb = input('Input an adverb: ')
test_adjective = input('Input the relevant adjective, e.g. apparently -> apparent: ')
emb_adv = emb[word2id[test_adverb]]
emb_adj = emb[word2id[test_adjective]]
dist = np.linalg.norm(emb_adv - emb_adj)
similarity = cosine_similarity(emb_adv, emb_adj)
print(f'd({test_adverb}, {test_adjective}) = {dist}')
print(f's({test_adverb}, {test_adjective}) = {similarity}')

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

