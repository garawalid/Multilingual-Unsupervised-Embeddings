# This script get the best dictionary from two word embedding using their mapping W
# This method is based on the first nearest neighbor and the similarity.
# n_words : is the number of best translation.
# You have to change the limit in load_word2vec_format in order to avoid memory error.

import numpy as np
import torch
from gensim.models import KeyedVectors
from numpy import linalg as LA
from sklearn.preprocessing import normalize

print 'loading mapping'
W = torch.load('dumped/debug/wiki/best_mapping.pth')
print 'loading data'
Y = KeyedVectors.load_word2vec_format('dumped/debug/wiki/vectors-fr.txt', limit=10000)
X = KeyedVectors.load_word2vec_format('dumped/debug/wiki/vectors-en.txt', limit=600)
print 'data loaded successfully'

Yt = Y.syn0
Yt = Yt.T
Y_normal = normalize(Yt, axis=1, norm='l2')


def translate(word):
    print ('processing :' + str(word))
    X_s = X.get_vector(word)
    WX_s = np.matmul(W, X_s)
    all_similarity = np.multiply(np.dot(WX_s, Y_normal), (1. / LA.norm(WX_s)))
    print(' similarity :')
    print np.sort(all_similarity)[-8:]
    k_ind = all_similarity.argsort()[-8:][::-1]
    print('translation of : ' + str(word))

    for p in k_ind:
        print Y.index2word[p]


def get_top_index(X, n):
    print "Get top index !"
    j_ind = X.argsort(axis=1)[:, -1:][:, -1]
    X_j = np.array([])
    count = 0
    for j in j_ind:
        X_j = np.append(X_j, X[count, j])
        count += 1
    i_ind = X_j.argsort()[-n:][::-1]
    j_ind = j_ind[i_ind]
    return i_ind, j_ind


def get_dict(X, n_words):
    print ('start dict')
    Xs = X.syn0
    Xs = Xs.T
    X_normal = normalize(Xs, axis=1, norm='l2')
    WXs = np.matmul(W, X_normal)
    similarity = np.matmul(WXs.T, Y_normal)
    i_ind, j_ind = get_top_index(similarity, n_words)
    for p in range(0, i_ind.shape[0]):
        print X.index2word[i_ind[p]]
        print Y.index2word[j_ind[p]]
        print ('similarity :' + str(similarity[i_ind[p], j_ind[p]]))
        print ('----')


n_words = 40
get_dict(X, n_words)
