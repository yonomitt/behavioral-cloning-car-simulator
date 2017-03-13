import pickle
import numpy as np
import tensorflow as tf

tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.visualize_util import plot


def conv_3_fc_3(dropout = [0.5, 0.5]):

    """This network has three convolution layers and three fully connected layers

    Parameters:
        dropout - list of dropout values for the 3 fully connected layers"""

    params = {
        'conv1': { 'filters': 8,  'size': 5 },
        'conv2': { 'filters': 16, 'size': 3 },
        'conv3': { 'filters': 32, 'size': 3 },
        'full4': { 'outputs': 556 },
        'full5': { 'outputs': 24 },
    }

    model = Sequential()

    # crop top 28 rows and bottom 12 rows from the images
    model.add(Cropping2D(cropping=((28, 12), (0, 0)), input_shape=(80, 160, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # layer 1: convolution. Input 40
    model.add(Convolution2D(8, 5, 5, border_mode='same', name='conv1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Activation('relu', name='act1'))
    model.add(Convolution2D(16, 3, 3, border_mode='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    model.add(Activation('relu', name='act2'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', name='conv3'))
    model.add(MaxPooling2D((2, 2), name='pool3'))
    model.add(Activation('relu', name='act3'))
    model.add(Flatten(name='flat'))
    model.add(Dense(556, name='fc4'))
    model.add(Dropout(dropout[0], name='drop4'))
    model.add(Activation('relu', name='act4'))
    model.add(Dense(24, name='fc5'))
    model.add(Dropout(dropout[1], name='drop5'))
    model.add(Activation('relu', name='act5'))
    model.add(Dense(1, name='out'))

    return model

if __name__ == '__main__':

    model = conv_3_fc_3(dropout=[0.2, 0.5])
    model.summary()
    plot(model, show_shapes=True, to_file='model.png')

    model.compile(loss='mse', optimizer='adam')
