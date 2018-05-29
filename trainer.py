"""This script is for training and evaluating a model."""

import sys
import os
import traceback
import numpy as np
from functools import partial
from utils import *
from punctuator import Punctuator
from bidirectional_gru_with_gru import BidirectionalGruWithGru

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam

EMBEDDING_FILE = 'data/glove.6B.50d.txt'
MODEL_FILE = 'data/model.json'
WEIGHTS_FILE = 'data/model.h5'
TEXT_FILE = 'data/training_text.txt'
BATCH = 100 
EPOCH = 1
DEV_SIZE = 1

def load_text_data(textfile):
    """Read a text file containing lines of text.
    Args:
        textfile: string representing a path name to a file
    Returns:
        list of words
    """
    words = []
    with open(textfile, 'r') as lines:
        for line in lines:
            words.extend(line.split())

    return words

def main():
    """Train a model using lines of text contained in a file
    and evaluates the model.
    """

    #read golve vecs
    words, word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(EMBEDDING_FILE)
    #create word embedding matrix
    embedding_matrix = create_emb_matrix(word_to_index, word_to_vec_map)
    print('shape of embedding_matrix:', embedding_matrix.shape)

    #load trainig text from a file
    utterances = load_text_data(TEXT_FILE)
    print(utterances[0])

    #create an instance of Punctutor and create training data
    punctuator = Punctuator(word_to_index, None)
    X, Y = punctuator.create_training_data(utterances, False)

    #if a model already exists, load the model
    if os.path.isfile(MODEL_FILE):
        punctuator.load_model(MODEL_FILE)
    else: 
        model = BidirectionalGruWithGru.create_model(
            input_shape=(X.shape[1], ), embedding_matrix=embedding_matrix,
            vocab_len=len(word_to_index), n_d1=128, n_d2=128, n_c=len(punctuator.labels))
        print(model.summary())
        punctuator.__model__ = model

    #if the model has been already trained, use the pre-trained weights
    if os.path.isfile(WEIGHTS_FILE): 
        punctuator.load_weights(WEIGHTS_FILE)

    #shuffle the training data
    shuffle(X,Y)
 
    denom_Y = Y.swapaxes(0,1).sum((0,1))
    print ('Summary of Y:', denom_Y)

    print('shape of X:', X.shape)
    print(X[0:10]) 
    print('shape of Y:', Y.shape)
    print(Y[0:10])

    #define optimizer and compile the model
    opt = Adam(lr=0.007, beta_1=0.9, beta_2=0.999, decay=0.01)
    punctuator.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    #split the training data into training set, test set, and dev set
    t_size = int(X.shape[0] * 0.9)
    train_X, train_Y = X[:t_size], Y[:t_size]
    test_X, test_Y = X[t_size:-DEV_SIZE], Y[t_size:-DEV_SIZE]
    dev_X, dev_Y = X[-DEV_SIZE:], Y[-DEV_SIZE:]

    print (train_Y.swapaxes(0,1).sum((0,1)))
    print (test_Y.swapaxes(0,1).sum((0,1)))

    #train the model
    punctuator.fit([train_X], train_Y, batch_size = BATCH,
               epochs=EPOCH)
    punctuator.save_model(MODEL_FILE)
    punctuator.save_weights(WEIGHTS_FILE)

    #evaluate the model on the dev set (or the test set)
    for i,example in enumerate(dev_X):
        prediction = punctuator.predict(example)
        punctuator.check_result(prediction, dev_Y[i])

    #manually evaluate the model on an example
    examples = ["good morning chairman who I saw and members of the committee it's my pleasure to be here today I'm Elizabeth Ackles director of the office of rate payer advocates and I appreciate the chance to present on oris key activities from 2017 I have a short presentation and I'm going to move through it really quickly because you've had a long morning already and be happy to answer any questions that you have"]
    for example in examples:
        words = example.split()
        x = punctuator.create_live_data(words)
        print x
        for s in x:
            print s
            prediction = punctuator.predict(s)
            result = punctuator.add_punctuation(prediction, words)
            print(result)

if __name__ == "__main__":
    main()
