# This script allows you to translate some words using the two word embedding and their mapping W
# words_2_translate : words to be translated.
# Hints: you can add limit to load_word2vec_format in order to reduce the usage of memory,
# but be aware, may be the word that you want to translate doesn't exist in the loaded word embedding

import numpy as np
import torch
from gensim.models import KeyedVectors
from numpy import linalg as LA
from sklearn.preprocessing import normalize

print 'loading mapping'
W = torch.load('dumped/debug/wiki/best_mapping.pth')

print 'loading data'
Y = KeyedVectors.load_word2vec_format('dumped/debug/wiki/vectors-fr.txt')
X = KeyedVectors.load_word2vec_format('dumped/debug/wiki/vectors-en.txt')
print 'data loaded successfully'
# Processing Y
Yt = Y.syn0
Yt = Yt.T

Y_normal = normalize(Yt, axis=1, norm='l2')


def translate(word):
    print ('processing :' + str(word))
    X_s = X.get_vector(word)
    WX_s = np.matmul(W, X_s)
    all_simliarity = np.multiply(np.dot(WX_s, Y_normal), (1. / LA.norm(WX_s)))
    print(' similarity :')
    print np.sort(all_simliarity)[-8:]
    k_ind = all_simliarity.argsort()[-8:][::-1]
    print('translation of : ' + str(word))

    for p in k_ind:
        print Y.index2word[p]


words_2_translate = ['in', 'a', 'one', 'bank', 'rate', 'more', 'new', 'march', 'year', 'send', 'like', 'will', 'happy',
                     'for', 'tow', 'house', 'car', 'she', 'door', 'the']

# Translating
for word_i in words_2_translate:
    translate(word_i)
