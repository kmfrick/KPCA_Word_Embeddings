from datasets import load_dataset
from torchnlp import word_to_vector
import torch
import subprocess
from sklearn.decomposition import PCA, KernelPCA

from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer


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


def get_word2vec(which="fasttext"):
    which = which.lower()
    if which == "fasttext":
        return word_to_vector.FastText()
    if which == "glove":
        return word_to_vector.Glove()
    if which == "skip_gram":
        return word_to_vector.CharNGram()


def embed(dataset, method):
    embedding_weights = torch.Tensor(dataset["train"].num_rows, method.dim)
    for i, token in enumerate(range(dataset["train"].num_rows)):
        # Here sentences are fed in, those probably have to be cut into words somehow
        embedding_weights[i] = method[dataset["train"][token]["text"]]

    return embedding_weights


def fit_with_kpca(data):
    kpca = KernelPCA(kernel="rbf")
    return kpca.fit_transform(data)


def get_vocabulary(dataset):
    trainer = WordLevelTrainer(vocab_size=30000)
    tokenizer = Tokenizer(WordLevel())
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train_from_iterator(dataset["text"], trainer)
    # TODO implement loading from this file instead of training
    tokenizer.save("wiki.json")

    return tokenizer.get_vocab()


def main():
    # get a dataset first - choose from "text", "news" and "german"
    dataset = get_dataset("text")
    # tokenize the dataset -> single words from sentences
    vocab = get_vocabulary(dataset["train"])
    # get a method for calculating a vector from words - choose from "fasttext", "glove" and "skip_gram"
    word2vec = get_word2vec("fasttext")
    weights = embed(vocab, word2vec)

    # TODO preprocess with KPCA
    # TODO check why this wants to store multiple TB - dim probably way too big
    # fit_with_kpca(...)

    # TODO compare normal embeddings to embeddings with preprocess


if __name__ == "__main__":
    main()
