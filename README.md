# Multilingual-Unsupervised-Embeddings
The main goal of this project is to share some insights and feedbacks to align two different words embedding and then translate some words from English to French using [MUSE](https://github.com/facebookresearch/MUSE) (unsupervised approach). 

First, we start building a word embedding from [The Reuters data set](http://www.nltk.org/nltk_data/) using [fastText](https://github.com/facebookresearch/fastText). Then we choose French [fastText Wikipedia embedding](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) as a target.

In the data directory, you will find the Reuters data set and their word embeddings. Note that we used 300 as a dimension.
After aligning these two words embedding, we found that the performance is very bad. The accuracy equals 0.13, based on 734 words using the 10-Nearest neighbor.

It was expected since the Reuters data set has only 33995 words and this method relay on the co-occurrence of words. You can find all the details about this in the [log file](dumped/debug/reuters/train_reuters).

So we decided to change the first embedding and replace it with wiki.en.vec. We kept the same target. The result was good, the accuracy was 0.786 based on 1500 words using 1-Nearest Neighbor. Using the  CSLS* metric leads to 0.822 accuracy. ([log file](dumped/debug/wiki/train_wiki))

CSLS* : CROSS-DOMAIN SIMILARITY LOCAL SCALING simply is a new metric developed to get a better mapping, [read more](https://arxiv.org/pdf/1710.04087.pdf)

After getting the mapping, we can translate some words using translate_wiki.py , and also it’s possible to get the best dictionary. These two approaches rely on the 1-nearest neighbor.

You can download embeddings of this experiment from [here](https://mega.nz/#F!FEMThQ5B!dYQD6R53zKQFXV14RLIVFA)
