import os.path

from datasets import load_dataset
import subprocess
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
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
import re

from itertools import chain

# Progress bar
from tqdm import tqdm

def get_huggingface_dataset():
        return load_dataset('wikitext', 'wikitext-103-raw-v1')

def get_vocabulary(dataset, vocab_size = 1000, reset=False):
    fname = f'text_{vocab_size}.json'
    if os.path.isfile(fname) and not reset:
        print(f'Loading from {fname}...')
        tokenizer = Tokenizer.from_file(fname)
    else:
        trainer = WordLevelTrainer(vocab_size=vocab_size)
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = Sequence([Punctuation(behavior='removed'), Whitespace()])
        regex_special_chars = Regex('[^\w\s]|[0-9]')
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents(), Replace(regex_special_chars, '')])
        print('Training tokenizer...')
        tokenizer.train_from_iterator(tqdm(dataset['text']), trainer)
        print(f'Saving to {fname}...')
        tokenizer.save(fname)
    return tokenizer

def get_ngrams(word, n = 3):
    return [''.join(g) for g in nltk.ngrams([c for c in word], n)]

def word_similarity(s1, s2):
    card1 = len(s1)
    card2 = len(s2)
    # This will avoid division by zero errors for words shorter than 3 characters
    # HOWEVER!
    # It's fine if we don't catch one- or two-letter words that are equal
    # We can fix that by setting the diagonal of S to ones in similarity_matrix
    # We *want* one- or two- letter words that are not exactly the same to have similarity 0, on the other hand
    if card1 == 0 and card2 == 0:
        return 0
    card_int = len(s1.intersection(s2))
    return 2 * card_int / (card1 + card2)



def similarity_vector(word_ngrams, vocab_ngrams, n = 3):
    m = len(vocab_ngrams)
    s = np.zeros(m)
    for i in range(0, m):
        s[i] = word_similarity(word_ngrams, vocab_ngrams[i])
    return s.reshape(1, -1)


def similarity_matrix(vocab_ngrams):
    m = len(vocab_ngrams)
    S = np.zeros([m, m])
    for i in tqdm(range(0, m)):
        S[i, :] = similarity_vector(vocab_ngrams[i], vocab_ngrams)
    # See word_similarity()
    np.fill_diagonal(S, np.ones(m))
    return S


# Sanity check: get the kernel matrix and the k-NN of a random word, check it makes sense
def knn_sanity_check(K, kpca, training_vocab, training_ngrams, n = 3, k = 5):
    # Run 5-NN on K
    print(K[10, :])
    print(K[280, :])
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(K)
    # Get neighbors of a test word
    test_word = input('test_word = ')
    print(f'test_word = {test_word}')
    test_word_ngrams = set(get_ngrams(test_word, n))
    s_new = similarity_vector(test_word_ngrams, training_ngrams)
    k_new = kpca.transform(s_new)
    neigh = nbrs.kneighbors(k_new, k)
    for i in neigh[1][0]:
        print(training_vocab[i])

class DataReaderHuggingFace(DataReader):

    def __init__(self, huggingface_dataset, min_count, mode='train'):
        self.sentences = []
        self.mode = mode
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        self.token_count = 0

        self.read_words(huggingface_dataset, min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words_untokenized(self, huggingface_dataset):
        word_frequency = dict()
        for line in tqdm(huggingface_dataset[self.mode]['text']):
            line = line.lower().strip()
            line = re.sub(' +', ' ', line)
            line = re.sub('[^\w\s]|[0-9]', '', line)
            if len(line) > 1:
                self.sentences.append(line)
                line = line.split()
                for word in line:
                    if len(word) > 1:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

        self.sentences_count = len(self.sentences)
        return word_frequency


    def read_words(self, huggingface_dataset, min_count):
        words = self.read_words_untokenized(huggingface_dataset)
        wid = 0
        for w, c in words.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print('Total embeddings: ' + str(len(self.word2id)))

class Word2vecDatasetHuggingFace(Word2vecDataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.rng = np.random.default_rng(50321)

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        sentence_id = self.rng.integers(self.data.sentences_count)
        words = self.data.sentences[sentence_id].split()

        word_ids = [self.data.word2id[w] for w in words if
                    w in self.data.word2id and self.rng.uniform() < self.data.discards[self.data.word2id[w]]]

        boundary = self.rng.integers(1, self.window_size)
        ret = [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]
        return ret


def train_word2vec(K, kpca, training_ngrams, huggingface_dataset, inject_kernel, n):
    min_count = 12
    print('Creating DataReader...')
    data = DataReaderHuggingFace(huggingface_dataset, min_count)
    # Inject K into word2vec training
    m = len(data.word2id)
    print(K.shape)
    p = K.shape[1]
    print(m, p)
    skip_gram_model = SkipGramModel(m, p)
    if (inject_kernel):
        print('Initializing word2vec embedding layer with KPCA matrix...')
        def get_similarity_vector(w):
            w_ngrams = set(get_ngrams(w, n))
            s_new = similarity_vector(w_ngrams, training_ngrams)
            return s_new
        S_new = np.array([get_similarity_vector(w) for w in tqdm(data.id2word.values())]).squeeze()
        K_new = kpca.transform(S_new)
        print(K_new.shape)
        print(skip_gram_model.u_embeddings.weight.data.shape)
        print(K_new[10, :])
        print(K_new[280, :])
        skip_gram_model.u_embeddings.weight.data = torch.nn.Parameter(torch.FloatTensor(K_new))
        print('Saving pure-KPCA embeddings...')
        skip_gram_model.save_embedding(data.id2word, f'Embeddings_{p}_InjectTrue_NoTraining.vec')


    # Train word2vec
    batch_size = 32
    initial_lr = 1e-3
    maxit = 3
    window_size = 5
    dataset = Word2vecDatasetHuggingFace(data, window_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, collate_fn=dataset.collate)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # See https://discuss.pytorch.org/t/valueerror-optimizer-got-an-empty-parameter-list-when-its-clearly-not-empty/102264/2
    for it in range(maxit):
        print(f'\n\n\nIteration: {it+1}')
        optimizer = torch.optim.SparseAdam(list(skip_gram_model.parameters()), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

        running_loss = 0.0
        with tqdm(dataloader) as tdata:
            for i, sample_batched in enumerate(tdata):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(device)
                    pos_v = sample_batched[1].to(device)
                    neg_v = sample_batched[2].to(device)

                    optimizer.zero_grad()
                    loss = skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    tdata.set_postfix_str(f' Loss: {running_loss}')
        output_file_name = f'Embeddings_{p}_Inject{inject_kernel}_{it+1}Epochs.vec'
        skip_gram_model.save_embedding(data.id2word, output_file_name)

def mse_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return mean_squared_error(X, X_preimage, squared=False) / np.mean(X)


def main():
    huggingface_dataset = get_huggingface_dataset()
    # tokenize the huggingface_dataset -> single words from sentences
    m = 1000
    tokenizer = get_vocabulary(huggingface_dataset['train'], vocab_size = m)
    training_vocab = tokenizer.get_vocab()
    # Remove words shorter than three characters, because they do not convey
    # sufficient information, therefore should not be included in the training similarity matrix
    training_vocab = [w for w in training_vocab if len(w) >= 3]
    m = len(training_vocab)
    n = 3
    training_ngrams = [set(get_ngrams(w, n)) for w in training_vocab]

    # Calculate similarity matrix
    print('Calculating similarity matrix...')
    S = similarity_matrix(training_ngrams)

    gamma_c = 1/m
    gridsearch = True
    if gridsearch:
        # Cosine reconstruction evaluation
        for gamma_i in range(-1, 4):
            gamma = gamma_c + gamma_i * 1e-3
            for n_components in [32, 64, 128, 256, 512]:
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
        # NMRSE evaluation
        param_grid = [{
            "gamma": np.linspace(1e-6, 1e-2, 20),
            'n_components': [32, 64, 128, 256, 512]
        }]

        kpca = KernelPCA(fit_inverse_transform = True, n_jobs = -1)
        clf = GridSearchCV(kpca, param_grid, cv = 5, scoring = mse_scorer)
        clf.fit(S)
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        exit()



    n_components = 128
    gamma = gamma_c

    print(f'Fitting a kernel PCA with {n_components} components and gamma = {gamma}')
    kpca = KernelPCA(kernel='rbf', n_components = n_components, gamma = gamma)
    K = kpca.fit_transform(S)

    # Quick sanity check using k nearest neighbors
    k = 5
    knn_sanity_check(K, kpca, training_vocab, training_ngrams, n, k)
    for inject_kernel in [True]:
        train_word2vec(K, kpca, training_ngrams, huggingface_dataset, inject_kernel, n)


if __name__ == '__main__':
    main()
