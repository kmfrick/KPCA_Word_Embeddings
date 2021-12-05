import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

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

# Get neighbors of a test word
test_word = input('Input a test word: ')
print(f'test_word = {test_word}')
test_id = word2id[test_word]
print(test_id)
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
print(f'd({test_adverb}, {test_adjective}) = {dist}')

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
