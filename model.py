import csv
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
import traceback

tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.visualize_util import plot


STEERING_CORRECTION = 0.2
VALIDATION_PCT = 0.2


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

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

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


def data_generator(samples, batch_size=128):

    """A generator method to provide the model with data during training"""

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                image = cv2.resize(image, (160, 80))
                angle = batch_sample[1]
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def gen_rel_img(base, path):

    """Helper function to return the image relative to the base directory

    Parameters:
        - path: path to the image

    Returns: string with the image file name realtive to the base directory"""

    if path.lower().startswith("c:"):
        path = path.split('\\')[-1]

    return os.path.join(base, "IMG", os.path.split(path.strip())[-1])


def read_samples(base_dirs):

    """Read the samples from CSV files

    Parameters:
        - base_dirs: list of directories containing the CSV files
        
    Returns: list of tuples containing the absolute path to the image and the normalized steering angle"""

    samples = []

    for base_dir in base_dirs:

        with open(os.path.join(base_dir, "driving_log.csv")) as f:
            log = [l.split(',') for l in f.read().split('\n')[1:-1]]

            # center image
            samples.extend([(gen_rel_img(base_dir, l[0]), float(l[3]) / 25.0) for l in log])
            # left image
            samples.extend([(gen_rel_img(base_dir, l[1]), (float(l[3]) + STEERING_CORRECTION) / 25.0) for l in log])
            # right image
            samples.extend([(gen_rel_img(base_dir, l[2]), (float(l[3]) - STEERING_CORRECTION) / 25.0) for l in log])

    return samples


if __name__ == '__main__':

    batch_size = 32
    nb_epoch = 5

    model = conv_3_fc_3(dropout=[0.2, 0.5])
    model.summary()

    exp_name = "{}.b{}.e{}".format(model.name, batch_size, nb_epoch)

    plot(model, show_shapes=True, to_file='results/model_{}.png'.format(exp_name))

    # get data
    samples = read_samples(['data/udacity'])

    n_samples = len(samples)
    n_valid = round(n_samples * VALIDATION_PCT)
    n_train = n_samples - n_valid

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:]

    train_generator = data_generator(train_samples, batch_size=batch_size)
    valid_generator = data_generator(valid_samples, batch_size=batch_size)

    model.compile(loss='mse', optimizer='adam')

    print("n_samples: {}".format(n_samples))
    print("n_train: {}".format(n_train))
    print("n_valid: {}".format(n_valid))

    history_object = model.fit_generator(train_generator, samples_per_epoch=n_train,
            validation_data=valid_generator, nb_val_samples=n_valid, nb_epoch=nb_epoch)

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('results/loss_{}.png'.format(exp_name))

