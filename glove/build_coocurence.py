# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
import numpy as np
import time
import logging
from scipy.sparse import lil_matrix
import cPickle as pickle
import json


path = '../data/text8'


class generateCoocur(object):

    def __init__(self, file_path=None):
        #self.coocurence = np.array([])
        self.vocab = {}
        self.vocab_i2w = {}
        # extractText object which is an iterator over wiki articles
        if file_path is not None:
            self.document = open(file_path, 'r')
        else:
            self.document = open(path, 'r')
        self.coocur_mat = []

    def gen_vocab(self):
        '''
        Generates vocabulary for entire corpus
        '''
        word_id = 0
        for page in self.document:
            doc_tokens = page.strip().split()
            # Find the vocab of the doc by finding number of uniq elem
            for token in doc_tokens:
                if token not in self.vocab:
                    self.vocab[token] = word_id
                    self.vocab_i2w[word_id] = token
                    word_id += 1

    def gen_coocur(self, window_size=3):
        '''
        Generates coocurrence matrix
        '''
        # Add padding to coocurence matrix to account for corner cases
        self.coocur_mat = lil_matrix((len((self.vocab)), len(self.vocab)))
        M = len(self.vocab)
        self.document.seek(0)
        for line in self.document:
            # TODO: shift this preprocessing to 'extract-text.py'
            tokens = line.strip().split()
            N = len(tokens)
            # To handle edge cases of context window
            for i in range(0, N):
                if i < window_size:
                    for j in xrange(0, i):
                        self.coocur_mat[
                            self.vocab[tokens[i]], self.vocab[tokens[j]]] += 1 / float(abs(i - j))
                        self.coocur_mat[
                            self.vocab[tokens[j]], self.vocab[tokens[i]]] += 1 / float(abs(i - j))
                elif i >= window_size:
                    for j in xrange((i - window_size), i):
                        self.coocur_mat[
                            self.vocab[tokens[i]], self.vocab[tokens[j]]] += 1 / float(abs(i - j))
                        self.coocur_mat[
                            self.vocab[tokens[j]], self.vocab[tokens[i]]] += 1 / float(abs(i - j))
        # Pickle the matrix
        with open("coocurence.mat", "w") as outfile:
            pickle.dump(self.coocur_mat, outfile, pickle.HIGHEST_PROTOCOL)

        with open("vocab.json", 'w') as f:
            json.dump(self.vocab, f)


if __name__ == "__main__":
    coocur = generateCoocur()
    t = time.time()
    coocur.gen_vocab()
    coocur.gen_coocur(5)
    print "Time taken is %f" % (time.time() - t)
