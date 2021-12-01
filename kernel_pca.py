import os.path

from datasets import load_dataset
from torchnlp import word_to_vector
import torch
import subprocess
from sklearn.decomposition import PCA, KernelPCA

from tokenizers import Regex
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace
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


def embed(tokenizer, method):
    """
    manipulate this function for other embeddings. Given a tokenizer this gets
    the vocabulary and using the method calculates a vector/weight for each entry
    """
    embedding_weights = torch.Tensor(len(tokenizer.get_vocab()), method.dim)
    for i, token in enumerate(range(len(tokenizer.get_vocab()))):
        embedding_weights[i] = method[tokenizer.decode([i])]

    return embedding_weights


def fit_with_kpca(data):
    kpca = KernelPCA(kernel="rbf")
    return kpca.fit_transform(data)


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


def main():
    # get a dataset first - choose from "text", "news" and "german"
    dataset = get_dataset("text")
    # tokenize the dataset -> single words from sentences
    tokenizer = get_vocabulary(dataset["train"])
    # get a method for calculating a vector from words - choose from "fasttext", "glove" and "skip_gram"
    word2vec = get_word2vec("fasttext")
    weights = embed(tokenizer, word2vec)

    # preprocess with KPCA
    # TODO it does some pca magic but I am not sure how to use this atm
    fitted = fit_with_kpca(weights)

    # TODO compare normal embeddings to embeddings with preprocess


if __name__ == "__main__":
    main()
