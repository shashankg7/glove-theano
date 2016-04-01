
import numpy as np
from scipy.spatial import distance
import json

embed_mat = np.load('lookup.npy')
d = np.sum(embed_mat ** 2, 1) ** 0.5
W_norm = (embed_mat.T / d).T
f = open('vocab.json')
vocab = json.load(f)
vocab_i2w = {v: k for k, v in vocab.items()}


def similarity(word1, word2):
    ind = vocab[word1]
    ind1 = vocab[word2]
    wordvec = W_norm[ind, :]
    wordvec1 = W_norm[ind1, :]
    sim = 1 - distance.cosine(wordvec, wordvec1)
    print sim


def evaluate(word):
    ind = vocab[word]
    wordvec = W_norm[ind, :]
    dist = [distance.cosine(wordvec, vec) for vec in W_norm]
    indices = np.argsort(dist)
    closest_words = [vocab_i2w[ind] for ind in indices]
    closest_words = closest_words[1:10]
    print closest_words

if __name__ == "__main__":
    print "Enter words and system will display similar words, enter 'exit' to exit"
    while True:
        inp = raw_input()
        evaluate(inp)
        if inp == "exit":
        	break
    # while True:
    #	inp1, inp2 = raw_input().split()
    #	similarity(inp1, inp2)
