from model_factory import ModelFactory

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute, Layer, RNN
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam
from keras import backend as K

import numpy as np

from utils import *

class DBNCell(Layer):
    def __init__(self, output_dim, input_dim, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        self.output_dim = output_dim
        self.output_size = output_dim
        self.state_size = output_dim

        super(DBNCell, self).__init__(trainable=trainable, name=name, **kwargs)
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

    def call(self, x, _):
        linear_transform = K.dot(x + K.squeeze(self.bv, axis=-1), self.w) + K.squeeze(self.bh, axis=-1)
        return K.relu(linear_transform), [linear_transform] 

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

	cell1 = DBNCell(1800, 900, trainable=False)
	cell2 = DBNCell(900, 1800, trainable=False)
	cell3 = DBNCell(450, 900, trainable=False)
	cell4 = DBNCell(225, 450, trainable=False)
	cell5 = DBNCell(112, 225, trainable=False)
	cell6 = DBNCell(50, 112, trainable=False)
	cell1.load_weights('data/weights_dense_1.h5')
	cell2.load_weights('data/weights_dense_2.h5')
	cell3.load_weights('data/weights_dense_3.h5')
	cell4.load_weights('data/weights_dense_4.h5')
	cell5.load_weights('data/weights_dense_5.h5')
	cell6.load_weights('data/weights_dense_6.h5')
        cells = [cell1, cell2, cell3, cell4, cell5, cell6]
        X = RNN(cells, return_sequences=True, trainable=False)(X_input)

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

