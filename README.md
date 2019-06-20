# Self-Supervised Neural Machine Translation

This is the code used for the paper *Self-Supervised Neural Machine Translation*, which describes a joint parallel data extraction and NMT training approach. It is based on a December 2018 copy of the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) repository. Be aware that it is therefore not up-to-date with current changes in the original OpenNMT-py code.

## Installation

1. Clone this repository and make sure you have [Anaconda Python](https://www.anaconda.com/distribution/) installed.

2. Create a virtual environment with Python 3.6 and activate
```
conda create -n comparableNMT python=3.6
conda activate comparableNMT
```

3. Install the following packages...
   * [Pytorch](https://pytorch.org/) with CUDA support (if available)
   * Torchtext version 0.3.1
   * Configargparse
   * Six

## Instructions

In order to train the system without any paralell data, you will need plenty of comparable data as well as pre-trained word embeddings. It is vital that the comparable data used for extraction uses the same [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt) (BPE) as the data that the word embeddings were trained on.

### Training unsupervised embeddings

1. Accumulate large amounts of monolingual data in your L1 and L2 languages.

2. Preprocess (e.g. using [Moses scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts)) and apply BPE encoding.

3. Train [Word2Vec](https://github.com/tmikolov/word2vec) monolingual embeddings for L1 as well as for L2 data.

4. Map monolingual embeddings into common space using [Vecmap](https://github.com/tmikolov/word2vec) and a seed dictionary of numerals.

5. Once the embeddings are mapped into a common space, merge the source and target embeddings into a single file (you'll need to write a small script for doing that).

### Prepare comparable corpus

1. Collect a comparable corpus, ensuring that an article in L1 talks about the same topic as the corresponding article in L2.

2. Preprocess data, such that the final BPE encoding is the same as the one used for the monolingual data that the embeddings were trained on.

3. Now, you should have a directory containing all your pre-processed comparable documents. For the system to know which ones are related to each other, create a `list-file`. In each line, it contains the absolute path to a document in L1 which should be mapped to a document in L2. Use a tab between the L1 and L2 document. As such:
```
/path/to/dogs.L1\t/path/to/dogs.L2\n
/path/to/cats.L1\t/path/to/cats.L2\n
...
```

4. Also, create a concatenated version of all the comparable documents. We will call these two files `corpus.L1` and `corpus.L2` for now. These are only needed to create the OpenNMT-py format vocabulary files.

### Create OpenNMT corpus files

1. Run preprocess.py on the `corpus.L1/L2` files. Do not forget to set the vocabulary size high enough to cover all words in the vocabulary. Also, it is *vital* to the system to have a shared vocabulary, so do not forget to set the -share_vocab tag. A sample run command can be found under `comparableNMT/run/examples/run_preproc.sh`

2. Convert embeddings into OpenNMT format using embeddings_to_torch.py in the main directory. A sample run can be found under `comparableNMT/run/examples/run_embeds.sh`

### Train

Use train.py to train the system. Note that not all options available for training in OpenNMT-py are compatible with this code. For example, training on more than one GPU is currently not possible and batch sizes need to be given in number of sentences and not tokens. The examples in `comparableNMT/run/examples/run_train_XXXX.sh` show how to train the LSTM or Transformer models described in the paper. For best performance, look at `comparableNMT/run/examples/run_train_tf_margP.sh`

#### Comparable training options

```
--comparable: use joint parallel data extraction and training (mandatory)
--comparable_data: path to list-file (mandatory)
--comp_log path to where extraction logs will be written (mandatory)
--comp_epochs: number of epochs to pass over the comparable data (mandatory)
--no_base: do not use any parallel data to pre-train your model (recommended)
--fast: reduces the search space to the first batch in each document (substantially faster)
--threshold: supply a similarity score that needs to be passed by candidates (not necessary if you are using dual representations)
--second: use medium permissibility (margR)
--sim_measure: similarity measure used (default: margin-based)
--k: number of k-nearest neighbors to use when scoring (default: 4)
--cove_type: the type of sentence representation creation to use (default: sum)
--representations: dual uses both representation types, while hidden-only and embed-only use C_h and C_e respectively. In case of not using dual, it is recommended to set a threshold. (default: dual)
--max_len: maximum length of sequences to train on
--write_dual: this will also write logs for those sentence pairs accepted by one of the two representations only (can be used with dual representations)
--no_swaps: do not perform random swaps in the src-tgt direction during training (not recommended)
--comp_example_limit: supply a maximum number of unique pairs to extract (not recommended)
```

Run `train.py -h` for more information.

## Quickstart

For a quick start, you can download and use the embeddings and comparable data used for the paper here. We also provide the original BPE used for pre-processing. (ADD LINK)

## Questions?

This code was used for research and is not stable. We can therefore not guarantee that any deviations from the example runs will work. In case you have problems running the example scripts, please open an issue.

In case you use this code or the pre-processed data or embeddings provided, please remember to cite the original paper:

```
@inproceedings{ruiter2019self-supervised,
  title={Self-Supervised Neural Machine Translation},
  author={Ruiter, Dana and Espa{\~{n}}a-Bonet, Cristina and van Genabith, Josef},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Volume 2: Short Papers},
  month     = {August},
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics}
}
```
The original [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) code this repository is based on can be cited via:

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```





