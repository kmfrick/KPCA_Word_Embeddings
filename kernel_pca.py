from datasets import load_dataset
from torchnlp import word_to_vector
import torch
import subprocess
from sklearn.decomposition import PCA, KernelPCA


def perl_preprocess(self, path):
    pipe = subprocess.Popen(
        ["perl", "./filter_datasets.pl", path], stdin=subprocess.PIPE
    )
    pipe.stdin.write(path)
    pipe.stdin.close()


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
        fasttext = word_to_vector.FastText()
    if which == "glove":
        glove = word_to_vector.Glove()
    if which == "skip_gram":
        skip_gram = word_to_vector.CharNGram()


def embed(dataset, method):
    embedding_weights = torch.Tensor(dataset.num_rows, method.dim)
    for i, token in enumerate(dataset.num_rows):
        embedding_weights[i] = method[token]


def fit_with_kpca(data):
    kpca = KernelPCA(kernel="rbf")
    return kpca.fit_transform(data)


def main():
    # get a dataset first - choose from "text", "news" and "german"
    dataset = get_dataset("text")
    # get a method for calculating a vector from words - choose from "fasttext", "glove" and "skip_gram"
    word2vec = get_word2vec("fasttext")
    embed(dataset, word2vec)

    # TODO preprocess with KPCA
    fit_with_kpca(...)

    # TODO compare normal embeddings to embeddings with preprocess


if __name__ == "__main__":
    main()
