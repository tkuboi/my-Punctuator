from model_factory import ModelFactory

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute, Layer
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam
from keras import backend as K

import numpy as np

from utils import *

class InputLayer(Layer):
    def __init__(self, output_dim, input_dim, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        self.output_dim = output_dim
        super(InputLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.w = self.add_weight(shape=(input_dim, output_dim),
                                 initializer='random_normal',
                                 trainable=trainable, name='w')
        self.bv = self.add_weight(shape=(input_dim,1),
                                  initializer='zeros',
                                  trainable=trainable, name='bv')
        self.bh = self.add_weight(shape=(output_dim,1),
                                  initializer='zeros',
                                  trainable=trainable, name='bh')

    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        weights = [f['param_{}'.format(p)] for p in range(f.attrs['nb_params'])]
        self.set_weights(weights)
        f.close()

    def call(self, x):
        linear_tranform = K.matmul(x + K.squeeze(self.bv), self.w) + K.squeeze(self.bh)
        return K.nn.relu(linear_tranform)

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

        #embedding_matrix = kwargs.get('embedding_matrix')
        vocab_len = kwargs.get('vocab_len')
        n_d1 = kwargs.get('n_d1')
        n_d2 = kwargs.get('n_d2')
        n_c = kwargs.get('n_c')

        #define input
        X_input = Input(shape = kwargs.get('input_shape'))
  
        #define and create mbedding layer
        #embedding_layer = Embedding(vocab_len + 1,
        #                        embedding_matrix.shape[1],
        #                        trainable=False)
        #embedding_layer.build((None,))
        #embedding_layer.set_weights([embedding_matrix])

        #add embedding layer
        #X = embedding_layer(X_input)

        dense1 = InputLayer(1800, activation='relu', trainable=False)
        dense2 = InputLayer(900, activation='relu', trainable=False)
        dense3 = InputLayer(450, activation='relu', trainable=False)
        dense4 = InputLayer(225, activation='relu', trainable=False)
        dense5 = InputLayer(112, activation='relu', trainable=False)
        dense6 = InputLayer(50, activation='relu', trainable=False)
        dense1.load_weights('weights_dense_1.h5')
        dense2.load_weights('weights_dense_2.h5')
        dense3.load_weights('weights_dense_3.h5')
        dense4.load_weights('weights_dense_4.h5')
        dense5.load_weights('weights_dense_5.h5')
        dense6.load_weights('weights_dense_6.h5')

        X = dense1(X_input)
        X = dense2(X)
        X = dense3(X)
        X = dense4(X)
        X = dense5(X)
        X = dense6(X)

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

