import os.path

from datasets import load_dataset
import subprocess
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from torchnlp import word_to_vector
import torch

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

def get_dataset(self, which="text", load_simmilar=True):
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

def get_word2vec(which, dim, is_include):
    which = which.lower()
    if which == "fasttext":
        return word_to_vector.FastText(dim=dim, is_include=is_include)
    if which == "glove":
        return word_to_vector.Glove(dim=dim, is_include=is_include)
    if which == "skip_gram":
        return word_to_vector.CharNGram(is_include=is_include)


def set_weights(model, embedding_weights):
    pass

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

def main():
    # get a dataset first - choose from "text", "news" and "german"
    dataset = get_dataset("text")
    # tokenize the dataset -> single words from sentences
    tokenizer = get_vocabulary(dataset["train"])
    vocab = tokenizer.get_vocab()
    training_vocab = thresh_vocab(vocab, 1)
    n = 3
    training_ngrams = [set(get_ngrams(w, n)) for w in training_vocab]

    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    S = similarity_matrix(training_ngrams)

    # Transform S using kernel PCA
    print(f"Fitting a kernel PCA...")
    kpca = KernelPCA(kernel="rbf")
    K = kpca.fit_transform(S)

    # Quick sanity check using k nearest neighbors
    knn_sanity_check(K, kpca, training_vocab, training_ngrams)

    # TODO Inject K into word2vec training

    # TODO compare normal embeddings to embeddings with preprocess


if __name__ == "__main__":
    main()
