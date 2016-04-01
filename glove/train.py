
from glove import glove
from build_coocurence import generateCoocur as gen_coocur
import os
import json
import cPickle as pickle
import numpy as np
import argparse
import logging
import time
import theano


class train_glove(object):

    def __init__(self):
        '''
        Define hyperparamters of the model. (Modify using arg parser.
        '''
        self.dim = 100
        self.n_epochs = 10
        self.minibatch_size = 100000
        self.path = '../data/text8'
        self.lr = 0.05
        self.context_size = 10

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("path", help="path to text file")
        parser.add_argument(
            "--dim", help="dimension of word vectors", type=int)
        parser.add_argument(
            "--epochs", help="no of epochs to run SGD", type=int)
        parser.add_argument(
            "--learning_rate", help="learning rate for gradient descent",
            type=float)
        parser.add_argument(
            "--mini_batchsize", help="size of mini-batch for training",
            type=int)
        parser.add_argument("--context_size",
                            help="context size for constructing coocurence matrix",
                            type=int)
        args = parser.parse_args()
        if args.path:
            self.path = args.path
        if args.dim:
            self.dim = args.dim
        if args.epochs:
            self.epochs = args.epochs
        if args.learning_rate:
            self.lr = args.learning_rate
        if args.context_size:
            self.context_size = args.context_size

    def train_minibatch(self):
        '''
        Train gloVe model
        '''
        logger = logging.getLogger('glove')
        logger.setLevel(logging.INFO)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('glove.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        t = time.time()
        coocur = gen_coocur(self.path)
        logger.info("Vocabulary constructed")
        if os.path.isfile('vocab.json'):
            f = open('vocab.json', 'r')
            coocur.vocab = json.load(f)
        else:
            coocur.gen_vocab()
        vocab_size = len(coocur.vocab)
        if os.path.isfile('coocurence.mat'):
            f = open('coocurence.mat', 'r')
            coocur.coocur_mat = pickle.load(f)
        else:
            coocur.gen_coocur(self.context_size)
        logger.info("Coocurence matrix constructed")
        # vocab and coocurence matrix is loaded/generated
        nnz = coocur.coocur_mat.nonzero()
        model = glove(vocab_size, self.dim, self.lr)
        # nnz has i,j indices of non-zero entries
        nz = np.zeros((nnz[0].shape[0], 2))
        nz[:, 0] = nnz[0]
        nz[:, 1] = nnz[1]
        np.random.shuffle(nz)
        logger.info("Training started")
        try:
            for epoch in xrange(self.n_epochs):
                for i in xrange(0, nnz[0].shape[0], self.minibatch_size):
                    indw = np.asarray(nz[i:(i + self.minibatch_size), 0], dtype=np.int32)
                    indw1 = np.asarray(nz[i:(i + self.minibatch_size), 1], dtype=np.int32)
                    batch_size = indw.shape[0]
                    X = np.asarray(coocur.coocur_mat[indw,
                                                     indw1].todense(), dtype=theano.config.floatX).reshape(batch_size, )
                    fX = np.zeros_like(X)
                    for i in xrange(0, X.shape[0]):
                        if X[i] > 100:
                            fX[i] = (X[i] / float(100.0)) ** 0.75
                        else:
                            fX[i] = 1.
                    X = np.log(X)
                    cost = model.sgd(indw, indw1, X, fX)
                    logger.info("Cost in epoch %d is %f" % (epoch, cost))
        except Exception as e:
            logger.debug("System encountered an error  %s" % e)
        logger.info("Training ended")
        model.save_params()
        logger.info("parameters saved")
        logger.info("Time to complete training is %f" % (time.time() - t))

if __name__ == "__main__":
    model = train_glove()
    model.arg_parser()
    model.train_minibatch()
