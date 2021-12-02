import os.path

from datasets import load_dataset
import subprocess
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from torchnlp import word_to_vector
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

# Progress bar
from tqdm import tqdm

def get_huggingface_dataset(self, which="text", load_simmilar=True):
    if not load_simmilar:
        if which == "text":
            url = "http://mattmahoney.net/dc/"
            data_files = {"train+test": url + "text8.zip"}
            return load_dataset("json", data_files=data_files, field="data")

        elif which == "news":
            url = "http://qwone.com/~jason/20Newsgroups/"
            data_files = {"train+test": url + "0news-19997.tar.gz"}
            return load_dataset("json", data_files=data_files, field="data")

        elif which == "german":
            url = "http://statmt.org/wmt14/training-monolingual-news-crawl/"
            data_files = {"train+test": url + "news.2013.de.shuffled.gz"}
            return load_dataset("json", data_files=data_files, field="data")

    else:
        # similar but not exact datasets
        if which == "text":
            return load_dataset("wikitext", "wikitext-103-raw-v1")
        elif which == "news":
            return load_dataset("multi_news")
        elif which == "german":
            return load_dataset("euronews", "de-sbb")


def get_vocabulary(dataset, which="text", reset=False):

    if os.path.isfile(f'{which}.json') and not reset:
        tokenizer = Tokenizer.from_file(f"{which}.json")
    else:
        trainer = WordLevelTrainer(vocab_size=1000)
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = Sequence([Punctuation(behavior="removed"), Whitespace()])
        regex_special_chars = Regex("[^\w\s]|[0-9]")
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents(), Replace(regex_special_chars, "")])

        tokenizer.train_from_iterator(dataset["text"], trainer)
        tokenizer.save("text.json")


    return tokenizer

def get_ngrams(word, n = 3):
    return [''.join(g) for g in nltk.ngrams([c for c in word], n)]

def word_similarity(s1, s2):
    card1 = len(s1)
    card2 = len(s2)
    card_int = len(s1.intersection(s2))
    return 2 * card_int / (card1 + card2)


def thresh_vocab(vocab, thresh_percent, n = 3):
    if thresh_percent > 1 or thresh_percent < 0:
        raise ValueError("Threshold must be between 0 and 1 (100%)")
    thresh = len(vocab) * thresh_percent
    words = Counter(vocab).most_common(thresh)
    words = zip(*words)
    words = list(words)[0]
    words = list(words)
    return [w for w in words if len(w) >=3]

def similarity_vector(word_ngrams, vocab_ngrams, n = 3):
    m = len(vocab_ngrams)
    s = np.zeros(m)
    for i in range(0, m):
        s[i] = word_similarity(word_ngrams, vocab_ngrams[i])
        #print(f"Similarity between {word} and {vocab_ngrams[i]} is {s[i]}")
    return s.reshape(1, -1)


def similarity_matrix(vocab_ngrams):
    m = len(vocab_ngrams)
    S = np.zeros([m, m])
    for i in range(0, m):
        S[i, :] = similarity_vector(vocab_ngrams[i], vocab_ngrams)
    return S



def kernel_vector(new_word, kpca, num_components=-1):
    if (num_components < 0):
        return kpca.transform(new_word)
    else:
        return kpca.transform(new_word)[:, :num_components]


# Sanity check: get the kernel matrix and the k-NN of a random word, check it makes sense
def knn_sanity_check(K, kpca, training_vocab, training_ngrams, n = 3, k = 5):
    # Run 5-NN on K
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(K)
    # Get neighbors of a test word
    test_word = training_vocab[np.random.randint(K.shape[0])]
    test_word_pos = training_vocab.index(test_word)
    print(f"test_word = {test_word}")
    test_word_ngrams = set(get_ngrams(test_word, n))
    s_new = similarity_vector(test_word_ngrams, training_ngrams)
    k_new = kernel_vector(s_new, kpca)
    neigh = nbrs.kneighbors(k_new, k)
    for i in neigh[1][0]:
        print(training_vocab[i])

class DataReaderHuggingFace(DataReader):

    def __init__(self, huggingface_dataset, min_count, mode="train"):
        self.huggingface_dataset = huggingface_dataset
        self.mode = mode
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()


    def read_words(self, min_count):
        word_frequency = dict()
        for line in self.huggingface_dataset[self.mode]:
            line = line['text'].split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")
        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

class Word2vecDatasetHuggingFace(Word2vecDataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            for line in self.data.huggingface_dataset[self.data.mode]:
                words = line['text'].split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

def main():
    # get a dataset first - choose from "text", "news" and "german"
    huggingface_dataset = get_huggingface_dataset("text")
    # tokenize the huggingface_dataset -> single words from sentences
    tokenizer = get_vocabulary(huggingface_dataset["train"])
    vocab = tokenizer.get_vocab()
    training_vocab = thresh_vocab(vocab, 1)
    n = 3
    training_ngrams = [set(get_ngrams(w, n)) for w in training_vocab]

    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    S = similarity_matrix(training_ngrams)
    print(S.shape)

    # Transform S using kernel PCA
    print(f"Fitting a kernel PCA...")
    kpca = KernelPCA(kernel="rbf")
    K = kpca.fit_transform(S)
    print(K.shape)
    print(K.shape[0])
    print(K.shape[1])

    # Quick sanity check using k nearest neighbors
    knn_sanity_check(K, kpca, training_vocab, training_ngrams)

    # Train word2vec
    min_count = 12
    print("Creating DataReader...")
    data = DataReaderHuggingFace(huggingface_dataset, min_count)
    # Inject K into word2vec training
    m = len(data.word2id)
    print(K.shape)
    p = K.shape[1]
    print(m, p)
    skip_gram_model = SkipGramModel(m, p)
    print("Injecting kernel matrix in word2vec embeddings...")
    for i, w in data.id2word.items():
        w_ngrams = set(get_ngrams(w, n))
        s_new = similarity_vector(w_ngrams, training_ngrams)
        k_new = kernel_vector(s_new, kpca)
        skip_gram_model.u_embeddings.weight.data[i, :] = torch.from_numpy(k_new)

    # Train word2vec
    batch_size = 32
    initial_lr = 1e-3
    maxit = 3
    window_size = 5
    dataset = Word2vecDatasetHuggingFace(data, window_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, collate_fn=dataset.collate)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    output_file_name = "out.vec"
    for it in range(maxit):
        print(f"\n\n\nIteration: {it+1}")
        optimizer = torch.optim.SparseAdam(skip_gram_model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

        running_loss = 0.0
        for i, sample_batched in enumerate(tqdm(dataloader)):

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
                if i > 0 and i % 500 == 0:
                    print(f" Loss: {running_loss}")

        skip_gram_model.save_embedding(data.id2word, output_file_name)

    # TODO compare normal embeddings to embeddings with preprocess


if __name__ == "__main__":
    main()
