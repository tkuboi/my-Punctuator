from model_factory import ModelFactory

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam
import numpy as np

from utils import *

class BidirectionalGruWithGru(ModelFactory):
    """ Factory class to create a model with
    bidirectional GRU layer with unidirectional GRU layer.
    """

    def __init__(self):
        pass

    @staticmethod
    def create_model(**kwargs):
        """ Function creating the model's graph in Keras.
        
        Argument:
        input_shape -- shape of the model's input data (using Keras conventions)
        embedding_matrix -- matrix to map word index to word embedding vector 
        vocab_len -- the size of vocaburary  
        n_d1 -- ouput dimension for 1st GRU layer
        n_d2 -- ouput dimension for 2nd GRU layer
        n_c -- ouput dimension for output layer

        Returns:
        model -- Keras model instance
        """

        embedding_matrix = kwargs.get('embedding_matrix')
        vocab_len = kwargs.get('vocab_len')
        n_d1 = kwargs.get('n_d1')
        n_d2 = kwargs.get('n_d2')
        n_c = kwargs.get('n_c')

        #define input
        X_input = Input(shape = kwargs.get('input_shape'))
  
        #define and create mbedding layer
        embedding_layer = Embedding(vocab_len + 1,
                                embedding_matrix.shape[1],
                                trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])

        #add embedding layer
        X = embedding_layer(X_input)

        #add bidirectional GRU layer
        X = Bidirectional(GRU(n_d1, return_sequences = True))(X)
        X = Dropout(0.5)(X)
        X = BatchNormalization()(X)

        #add another GRU layer (unidirectional)
        X = GRU(n_d2, return_sequences = True)(X)
        X = Dropout(0.5)(X)
        X = BatchNormalization()(X)

        #get output for each time slot
        outputs = TimeDistributed(Dense(n_c, activation=softmax2))(X)

        #create and return keras model instance
        return Model(inputs=[X_input],outputs=outputs)

