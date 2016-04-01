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

# Evaluation

To evaluate the model, run following command :

> python utils.py

This will prompt for a string input. System will compare the word vector of the input string with all word vectors and return the closest 10 words. This is the popular analogy task which is used for evaluating word embeddings.

# Logging

Program automatically logs the progress of training in a log file in the current directory and currently no message is displayed on console output. 

It is advised to run the program with screen utility in linux.

# Future work 

Data handling part is pretty slow. Populating coocurence matrix takes lot of time. If anyone can fix it, please feel free to send a PR.
