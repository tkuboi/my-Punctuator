import sys
import traceback
import numpy as np

from functools import partial
from utils import *

from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam
from bidirectional_gru_with_gru import DBNCell

UTTERANCE_SIZE = 200
LABELS = ['NOP','INIT',',','.','?']

class Punctuator:
    """A class which incorporates a Keras model and methods to work on the model.
    The class also includes a function to create labeled data, evaluating the model
    , and adding punctuations to text.
    The class is designed to work on English text.
    """
 
    def __init__(self, words_to_index, model):
        """Constructor
        Args:
            words_to_index: a dictionary mapping words to index
            model: a Keras model
        """
        self.size = UTTERANCE_SIZE
        self.labels = LABELS
        self.words_to_index = words_to_index
        self.__model__ = model
        self.score = {"correct":0, "wrong":0, "FP-punc":0, "FN-punc":0}

    def reset_score(self):
        self.score = {"correct":0, "wrong":0, "FP-punc":0, "FN-punc":0}

    def compile(self, opt, **kwargs):
        """Compile a Keras model with specified options
        Args:
            opt: optimizer
            **kwargs: a dictionary for other options
        """
        self.__model__.compile(opt, **kwargs) 

    def fit(self, train_X, train_Y, batch_size=10, epochs=1):
        """ calls fit method on the Keras model with options
        Args:
            train_X: training data
            train_Y: labels for the trainig data
            batch_size: batch size
            epochs: the number of epochs
        """
        self.__model__.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs)

    def save_model(self, filename):
        """Save a Keras model to a json file
        Args:
            filename: a path name to a file to save a model in
        """
        save_model(self.__model__, filename)

    def save_weights(self, filename):
        """Save weights to a file
        Args:
            filename: a path name to a file to save weights in
        """
        self.__model__.save_weights(filename)

    def load_model(self, filename):
        """Load a Keras model from a json file
        Args:
            filename: a path name to a file to load a model from
        """
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.__model__ = model_from_json(loaded_model_json, custom_objects={'DBNCell': DBNCell, 'softmax2': softmax2})

    def load_weights(self, filename):
        """Load weights from a file
        Args:
            filename: a path name to a file to load weights from
        """
        self.__model__.load_weights(filename)

    def evaluate(self, test_X, test_Y, batch_size=10, epochs=1):
        """Evaluate the model by calling the Keras model's evaluate method
        Args:
            test_X:
            test_Y:
            batch_size:
            epochs:
        Returns:
           Scalar test loss
        """ 
        return self.__model__.evaluate(
            test_X, test_Y, batch_size=batch_size, epochs=epochs)

    def predict(self, x):
        """Make a prediction by calling the Keras model's predict method
        Args:
            x: a list of index to word
        Returns:
            an array of confidence value per label per word index 
        """
        return self.__model__.predict([np.array([x])])

    def label_data(self, i, j, t, Y, word, eos, last_init):
        """Annotate a word with a label
           This is a private helper function. 
        Args:
            i: index to i th example
            j: index to j th word
            t: index to t th time step
            Y: an array of one hot vector of label per word per example
            word: string representing j th word
            eos: boolean value indicating if the end of sentence or not
            last_init: the time index of the start of the last sentence
        Returns:
            word: string for the word with punctuations removed and lower cased
            eos: boolean value indicating if the end of sentence or not
            last_init: the time index of the start of the last sentence  
        """

        word = word.lower().strip()

        if '.' in word and 'Mr.' not in word:
            Y[i][j][self.labels.index('.')] = 1
            word = word.replace('.', '')
            eos = True
        elif '?' in word:
            Y[i][j][self.labels.index('?')] = 1
            word = word.replace('?', '')
            eos = True
        elif ',' in word and word.index(',') == len(word) - 1:
            Y[i][j][self.labels.index(',')] = 1
            word = word.replace(',', '')
        elif j == 0 or eos:
            Y[i][j][self.labels.index('INIT')] = 1
            last_init = t
            eos = False
        else:
            Y[i][j][self.labels.index('NOP')] = 1

        return word, eos, last_init

    def create_data(self, i, j, X, word):
        """A private helper function to
          convert word to index and store it in 2D list
        Args:
            i: index for i th example
            j: index for j th word
            X: 2D list of index to word
            word: the j th word
        """ 
        word = word.lower().strip()
        idx = self.words_to_index.get(word)
        if idx is None:
            X[i][j] = self.words_to_index.get('[UNKNOWN]')
        else:
            X[i][j] = idx

    def create_data2(self, i, j, X, word):
        """A private helper function to
          convert word to index and store it in 2D list
        Args:
            i: index for i th example
            j: index for j th word
            X: 2D list of index to word
            word: the j th word
        """ 
        word = word.lower().strip().replace('\n', '')
        X[i][j][:] = to_vector(word) 

    def create_training_data(self, words, rewind_to_head=True):
        """Create training data
        Args:
            words: a list of word
            rewind_to_head: a boolean value specifying whether or not
                to rewind to the head of sentence when the end of one example
                is reached.
        Returns:
            a 2D list of index to word, a 2D list of labels
        """ 
        m = int(len(words) / self.size + 1) * 10
        Y = np.zeros((m, self.size, len(self.labels)))
        X = np.zeros((m, self.size, 900))

        i = 0
        j = 0
        t = 0
        eos = False
        last_init = 0
        while t < len(words):
            if j >= self.size:
                j = 0
                i += 1
                if rewind_to_head:
                    t = last_init
                if i >= m:
                    break

            word = words[t]
            word, eos, last_init = self.label_data(i, j, t, Y, word, eos, last_init)
            self.create_data2(i, j, X, word)

            t += 1
            j += 1

        if i < m and j < self.size:
            while j < self.size:
               Y[i][j][self.labels.index('NOP')] = 1
               j += 1
        i += 1

        return np.asarray(X[:i]), np.asarray(Y[:i])

    def create_live_data(self, words):
        """Create data to be fed to the predictor
        Args:
            words: a list of word
        Returns:
            a 2D list of index to word
        """
        m = int(len(words) / self.size + 1)
        Y = np.zeros((m, self.size, len(self.labels)))
        X = np.zeros((m, self.size, 900))

        i = 0
        j = 0
        t = 0
        while t < len(words):
            if j >= self.size:
                j = 0
                i += 1
                if i >= m:
                    break

            word = words[t]
            self.create_data2(i, j, X, word)

            t += 1
            j += 1

        return X

    def punctuate(self, utterance):
        """Add punctuations to text
        Args:
            utterance: a string text
        returns:
            A list of strings of text with punctuations added
        """
        results = []
        words = utterance.split()
        x = self.create_live_data(words)
        for s in x:
            prediction = self.predict(s)
            result = self.add_punctuation(prediction, words)
            results.append(result)
        return " ".join(results)

    def add_punctuation(self, prediction, words):
        """A private helper function to add punctuations to text
        Args:
            prediction: array of vector of confidence values
            words: a list of word
        returns:
            A string of text with punctuations added
        """
        y_hat = np.argmax(prediction[0], axis = -1)
        sentence = []
        for i,w in enumerate(words):
            if self.labels[y_hat[i]] == 'INIT':
                sentence.append(w[0].upper() + w[1:])
            elif self.labels[y_hat[i]] != 'NOP':
                sentence.append(w + self.labels[y_hat[i]])
            else:
                sentence.append(w)
        return ' '.join(sentence)

    def check_result(self, prediction, dev_y):
        """A helper function to check the prediction results
            against predetermined labels 
        Args:
            prediction: array of vector of confidence values
            dev_y: a list of labels 
        """
        y_hat = np.argmax(prediction[0], axis = -1)
        y = np.argmax(dev_y, axis = -1)
        print '-'*60
        print("Truth <=> Prediction")
        for i,v in enumerate(y):
            print("%s <=> %s" % (self.labels[v], self.labels[y_hat[i]]))
            if self.labels[v] == self.labels[y_hat[i]]:
                self.score['correct'] += 1
            else:
                self.score['wrong'] += 1
            if v != 0 and y_hat[i] == 0: self.score['FN-punc'] += 1
            if v == 0 and y_hat[i] != 0: self.score['FP-punc'] += 1

        print(self.score)
        print '-'*60

