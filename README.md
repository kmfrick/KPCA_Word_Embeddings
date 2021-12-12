---
title: Kernel PCA for Word Embeddings
author: Kevin Michael Frick, Sonia Petrini, Patrick Lutz
---

# Requirements to run experiments

- `python>=3.8`
- `datasets` from [huggingface](huggingface.co)
- `tokenizers` from [huggingface](huggingface.co)
- `nltk`
- `numpy`
- `scikit-learn`
- `scipy`
- `torch`
- `tqdm`
- [SentEval](https://github.com/facebookresearch/SentEval) which should be in a folder called `SentEval` in the same folder as `evaluation.py` 
- [PyTorch Word2vec](https://github.com/Andras7/word2vec-pytorch) which should be in a folder called `torchw2vec` in the same folder as `kernel_pca.py` 

# How to install prerequisites

```
pip install -r requirements.txt
git submodule update --init --recursive
```

# How to perform grid search for ex-ante KPCA

Set the `gridsearch` variable to True in `kernel_pca.py` to perform grid search of the `gamma` parameter and of KPCA dimensionality.
 
# How to perform grid search for ex-post KPCA

This should be run after computing embeddings with `InjectFalse`.
Set the `gridsearch` and `reduce_with_kpca` variables to True in `evaluation.py` to perform grid search of the `gamma` parameter and of KPCA dimensionality.

# How to compute embeddings

1. Set the `gridsearch` variable to False in `kernel_pca.py`.
2. In the same file, set the `m` and `n_components` variables to desired training vocabulary size and embedding dimensionality, respectively.
3. Set the desired hyperparameters and number of training epochs in the function `train_word2vec()`, then run `python3 kernel_pca.py`.

The training will be done automatically with and without injection of knowledge for `maxit` epochs.
One epoch takes around three hours on a 2020 MacBook Pro 13".

The script will output a number of files equal to twice the number of epochs, called `Embeddings_P_InjectX_NEpochs.vec`, where `P` is the dimensionality of the embeddings, X is `True` if a KPCA matrix was injected and `False` if it was not, and `N` is the number of epochs the model was trained for.

# How to perform evaluation of ex-ante KPCA

Set the `gridsearch` and `reduce_with_kpca` variables to False in `evaluation.py`.
Run `python3 evaluation.py` and you will be presented with a list of files in the current directory.
Select the `.vec` files that have been output by a previous run of `kernel_pca.py`.
For each file selected, the script will output:

- the Spearman correlation coefficient with state-of-the-art embeddings
- the five nearest neighbors of "history", "moscow", "linguistically", "statistics", "firecracker"
- the squared cosine distance between "quickly, quick", "statistically, statistical", "calmly, calm", "thoroughly, thorough"

# How to perform evaluation of ex-post KPCA

Set the `gridsearch` variable to False and the `reduce_with_kpca` variable to True in `evaluation.py`.
In the same file, set the `n_components` variable to the desired embedding dimensionality.
Then proceed as before.
