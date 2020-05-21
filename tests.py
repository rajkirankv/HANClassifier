import numpy as np
import pandas as pd
# import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SENT_LENGTH = 100
# MAX_SENT_LENGTH = 10
# MAX_SENTS = 15
MAX_SENTS = 2
# MAX_NB_WORDS = 20000
MAX_NB_WORDS = 20
EMBEDDING_DIM = 100
# EMBEDDING_DIM = 10
VALIDATION_SPLIT = 0.2


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        # self.W = self.init((input_shape[-1],))
        self.W = K.variable(self.init((input_shape[-1],)))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], input_shape[-1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def test_attention():
    attn = AttLayer()
    input_shape = (200, 200, 200)
    attn.build(input_shape)
    print()


# test_attention()
str = '1.278/^5'
not_numbers = re.compile(r'[^0-9]')
str_clean = re.sub(not_numbers, '', str)
print(str_clean)