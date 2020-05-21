'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''
# from __future__ import print_function

import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, GRU
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.engine.topology import Layer
from keras import initializers
from keras import backend


vocabulary_size = 20000
# cut texts after this number of words
# (among top max_features most common words)
max_sequence_length = 100
batch_size = 50


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = backend.tanh(backend.dot(x, self.W))
        ai = backend.exp(eij)
        weights = ai / backend.sum(ai, axis=1).dimshuffle(0, 'x')
        # weighted_input = x * weights.dimshuffle(0, 'x')
        weighted_input = x * weights
        return weighted_input.sum(axis=1)

    def get_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_imdb_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    num_classes = np.max(y_train) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape, ' y_train shape: ', y_train.shape)
    print('x_test shape:', x_test.shape, ' y_test shape: ', y_test.shape)

    return (x_train, y_train), (x_test, y_test)


def save_model(model):
    from keras.utils import plot_model
    model.summary()
    plot_model(model, to_file='model.png')


def build_model():
    # seq_input = Input(shape=(max_sequence_length,), dtype='int32')
    # embedding_layer = Embedding(vocabulary_size, 100, input_length=max_sequence_length)(seq_input)
    # lstm = Bidirectional(LSTM(100))(embedding_layer)
    # dropout = Dropout(0.5)(lstm)
    # preds = Dense(2, activation='softmax')(dropout)
    # model = Model(seq_input, preds)
    # model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
    # return model

    # Attention Layer and GRU
    seq_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedding_layer = Embedding(vocabulary_size, 100, input_length=max_sequence_length)(seq_input)
    gru = Bidirectional(GRU(100, return_sequences=True))(embedding_layer)
    l_att = AttentionLayer()(gru)
    # preds = Dense(2, activation='softmax')(l_att)
    # preds = Dense((batch_size*max_sequence_length, 2), activation='softmax')(l_att)
    preds = Dense(2, activation='softmax')(l_att)
    
    model = Model(seq_input, preds)
    model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
    return model



(x_train, y_train), (x_test, y_test) = get_imdb_data()
model = build_model()
save_model(model)
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=[x_test, y_test])
save_model(model)