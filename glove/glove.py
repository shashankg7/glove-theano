'''
Defines glove model and performs one mini-batch SGD update in theano.
'''

from theano import tensor as T
import theano
import numpy as np


class glove(object):

    def __init__(self, vocab_size, dim, lr=0.5):
        W = np.asarray(np.random.rand(vocab_size, dim),
                       dtype=theano.config.floatX) / float(dim)
        W1 = np.asarray((np.random.rand(vocab_size, dim)),
                        dtype=theano.config.floatX) / float(dim)
        self.W = theano.shared(W, name='W', borrow=True)
        self.W1 = theano.shared(W1, name='W1', borrow=True)
        gW = np.asarray(np.ones((vocab_size, dim)), dtype=theano.config.floatX)
        gW1 = np.asarray(
            np.ones((vocab_size, dim)), dtype=theano.config.floatX)
        self.gW = theano.shared(gW, name='gW', borrow=True)
        self.gW1 = theano.shared(gW1, name='gW1', borrow=True)
        X = T.vector()
        fX = T.vector()
        ind_W = T.ivector()
        ind_W1 = T.ivector()
        w = self.W[ind_W, :]
        w1 = self.W1[ind_W1, :]
        cost = T.sum(fX * ((T.sum(w * w1, axis=1) - X) ** 2))
        grad = T.clip(T.grad(cost, [w, w1]), -5.0, 5.0)
        updates1 = [(self.gW, T.inc_subtensor(self.gW[ind_W, :],
                                              grad[0] ** 2))]
        updates2 = [(self.gW1, T.inc_subtensor(self.gW1[ind_W1, :],
                                               grad[1] ** 2))]
        updates3 = [(self.W, T.inc_subtensor(self.W[ind_W, :],
                                             - (lr / T.sqrt(self.gW[ind_W, :])) *
                                             grad[0]))]
        updates4 = [(self.W1, T.inc_subtensor(self.W1[ind_W1, :],
                                              - (lr / T.sqrt(self.gW1[ind_W1, :])) *
                                              grad[1]))]
        updates = updates1 + updates2 + updates3 + updates4
        self.cost_fn = theano.function(
            inputs=[ind_W, ind_W1, X, fX], outputs=cost, updates=updates)

    def sgd(self, indw, indw1, X, fX):
        '''
        Performs one iteration of SGD.
        '''
        return self.cost_fn(indw, indw1, X, fX)

    def save_params(self):
        '''
        Saves the word embedding lookup matrix to file.
        '''
        W = self.W.get_value() + self.W1.get_value()
        np.save('lookup', W)
