"""This file contains utility functions.
Some functions in this file are from Andrew Ng's Deep Learning courses on Coursera
    and stack overflow. 
"""
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.models import model_from_json

import keras.backend as K

from urllib2 import Request as request
from itertools import product
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

def read_glove_vecs(glove_file):
    """read glove vecs word embedding matrix from a file

    Copied from Andrew Ng's Deep Learning courses on Coursera
    and modified

    Args:
        glove_file: a path name to a file
    Returns:
        words: a list of words
        words_to_index: a dictionary for words to index mapping
        index_to_words: a dictionary for index to words mapping
        word_to_vec_map: a dictionary to map a word to a vector
    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    words_to_index['[UNKNOWN]'] = i
    index_to_words[i] = '[UNKNOWN]'
    return words, words_to_index, index_to_words, word_to_vec_map            

def softmax(x):
    """Compute softmax values for each sets of scores in x.

    Copied from Andrew Ng's Deep Learning courses on Coursera
    """ 

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax2(x, axis=1):
    """Softmax activation function.

    Copied from Andrew Ng's Deep Learning courses on Coursera

    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

def is_number(word):
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for num in numbers:
        if num in word:
            return True
    return False

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def save_model(model, filename):
    # serialize model to JSON
    with open(filename, "w") as json_file:
        json_file.write(model.to_json())

def load_model(filename):
    # load json and create model
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)

def create_emb_matrix(word_to_index, word_to_vec_map):
    #create word embedding matrix -- index : embedding vector
    emb_matrix = np.zeros((len(word_to_index)+1, len(word_to_vec_map.get('word'))))
    for word, index in word_to_index.items():
        vec = word_to_vec_map.get(word)
        if vec is not None:
            emb_matrix[index, :] = vec
    return emb_matrix

def shuffle(X,Y):
    #shuffle examples and labels arrays together 
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
