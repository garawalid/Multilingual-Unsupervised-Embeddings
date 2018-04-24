# Multilingual-Unsupervised-Embeddings
The main goal of this project is to align two different word embedding and then translate some words from english to french using MUSE and the unsupervised approach.

So, first we start with building word embedding from The Reuters data set, so we used fastText. Then we choose french fastText Wikipedia embedding as a target.

In the data directory, you find The Reuters data set and their word embeddings, we used 300 as dimension.
After align these two word embeddings we found that the performance is very bad. The accuracy is about 0.13 tested on 734 words using k-Nearest neighbor at k = 10.
That’s expected since the reuters data set have only 33995 words and this method relay on the co-occurence of words. You can find all details about this in the log file (/dumped/data/reuters/train_reuters)
So, we decided to change the first embedding and replaced by wiki.en.vec and we kept the same target. We found a impressive result, the accuracy was 0.786 tested on 1500 words using K-Nearest Neighbor at k = 1. Even more by changing the metric to CSLS* , the accuracy was 0.822. (log file)
CSLS* : CROSS-DOMAIN SIMILARITY LOCAL SCALING simply is a new metric developed to get a better mapping, read more

After getting the mapping, we can translate some words using translate_wiki.py , and also it’s possible to get the best dictionary. These two approach relay on the k-nearest neighbor at k = 1. 
