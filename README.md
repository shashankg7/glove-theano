# GloVe Word Embedding model in Theano

Word Embedding model GloVe's implementation in theano. 

# Requirements

1. Theano
2. Numpy
3. Scipy

# Running GloVe

To train glove model on text corpus put the data file in the data folder in parent folder. Currently text8 corpus (wikipedia's first 1B characters) is present for demo purpose. 

To run the program :

> python train.py --help

It lists the arguments which could be passed for training. All paramters except text path are optional and are initialized with default values.

# Future work 

Data handling part is pretty slow. Populating coocurence matrix takes lot of time. If anyone can fix it, please feel free to send a PR.
