import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tqdm import tqdm

word2id = dict()
id2word = dict()

i = -1

with open('Embeddings_128_InjectFalse_text.vec') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        # Skip first line
        if i < 0:
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
print('Input a text word: ')
test_word = input()
print(f'test_word = {test_word}')
test_id = word2id[test_word]
print(test_id)
test_embedding = emb[test_id]
# Run 5-NN on K
k = 5
nbrs = NearestNeighbors(n_neighbors=k)
nbrs.fit(emb)
neigh = nbrs.kneighbors(test_embedding.reshape(1, -1), k+1)
neigh_list = neigh[1][0][1:]
print(neigh_list)


# Use PCA to plot 5-NN in 2D
pca = PCA(
    n_components=2,
    random_state=50321,
)
print('Fitting PCA...')
pos = pca.fit_transform(emb)
print(pos.shape)

fig = plt.figure(1)
plt.scatter(pos[neigh_list, 0], pos[neigh_list, 1], color='turquoise')
plt.scatter(pos[test_id, 0], pos[test_id, 1], color='red')

for j, xy in zip(neigh_list, pos[neigh_list, :]):
    plt.annotate(id2word[j], xy = xy, textcoords='data')

plt.title(f'Neighbors of {test_word}')

plt.show()
