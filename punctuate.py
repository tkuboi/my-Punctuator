"""This file contains a script to load a model and weights and test adding punctuations
to example text.
""" 
import sys
import traceback
import MySQLdb
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
UTTERANCE_SIZE = 200

def main():

    words, word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(EMBEDDING_FILE)


    punctuator = Punctuator(word_to_index, None)

    punctuator.load_model(MODEL_FILE)
    punctuator.load_weights(WEIGHTS_FILE)

    examples = ["good morning chairman who I saw and members of the committee it's my pleasure to be here today I'm Elizabeth Ackles director of the office of rate payer advocates and I appreciate the chance to present on oris key activities from 2017 I have a short presentation and I'm going to move through it really quickly because you've had a long morning already and be happy to answer any questions that you have"]
    for example in examples:
        words = example.split()
        x = punctuator.create_live_data(words)
        for s in x:
            prediction = punctuator.predict(s)
            result = punctuator.add_punctuation(prediction, words)
            print(result)

if __name__ == "__main__":
    main()
