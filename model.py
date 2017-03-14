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
from keras.callbacks import ModelCheckpoint
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

    # layer 1: convolution + max pooling. Input 40x160x3. Output 20x80x8
    model.add(Convolution2D(8, 5, 5, border_mode='same', name='conv1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Activation('relu', name='act1'))

    # layer 2: convolution = max pooling. Input 20x80x8. Output 10x40x16
    model.add(Convolution2D(16, 3, 3, border_mode='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    model.add(Activation('relu', name='act2'))

    # layer 3: convolution = max pooling. Input 10x40x16. Output 5x20x32
    model.add(Convolution2D(32, 3, 3, border_mode='same', name='conv3'))
    model.add(MaxPooling2D((2, 2), name='pool3'))
    model.add(Activation('relu', name='act3'))

    # flatten: Input 5x20x32. Output 3200
    model.add(Flatten(name='flat'))

    # layer 4: fully connected + dropout. Input 3200. Output 556
    model.add(Dense(556, name='fc4'))
    model.add(Dropout(dropout[0], name='drop4'))
    model.add(Activation('relu', name='act4'))

    # layer 5: fully connected + dropout. Input 556. Output 24
    model.add(Dense(24, name='fc5'))
    model.add(Dropout(dropout[1], name='drop5'))
    model.add(Activation('relu', name='act5'))

    # layer 6: fully connected. Input 24. Output 1.
    model.add(Dense(1, name='out'))

    return model


def end_to_end_nvidia(dropout = []):

    """This model attempts to mimic the model by NVIDIA in their paper End to End Learning for Self-Driving
    Cars:

    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"""

    params = {
        'conv1': { 'filters': 24, 'size': 5 },
        'conv2': { 'filters': 36, 'size': 5 },
        'conv3': { 'filters': 48, 'size': 5 },
        'conv4': { 'filters': 64, 'size': 3 },
        'conv5': { 'filters': 64, 'size': 3 },
        'full6': { 'outputs': 100 },
        'full7': { 'outputs': 50 },
        'full8': { 'outputs': 10 },
    }

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

    # crop top 28 rows and bottom 12 rows from the images
    model.add(Cropping2D(cropping=((28, 12), (0, 0)), input_shape=(80, 160, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # layer 1: convolution. Input 40x160x3. Output 36x156x24
    model.add(Convolution2D(24, 5, 5, border_mode='valid', name='conv1'))
    model.add(Activation('relu', name='act1'))

    # layer 2: convolution + max pooling. Input 36x156x24. Output 16x76x36
    model.add(Convolution2D(36, 5, 5, border_mode='valid', name='conv2'))
    model.add(MaxPooling2D((2, 2), border_mode='valid', name='pool2'))
    model.add(Activation('relu', name='act2'))

    # layer 3: convolution + max pooling. Input 16x76x36. Output 6x36x48
    model.add(Convolution2D(48, 5, 5, border_mode='valid', name='conv3'))
    model.add(MaxPooling2D((2, 2), border_mode='valid', name='pool3'))
    model.add(Activation('relu', name='act3'))

    # layer 4: convolution. Input 6x36x48. Output 4x34x64
    model.add(Convolution2D(64, 3, 3, border_mode='valid', name='conv4'))
    model.add(Activation('relu', name='act4'))

    # layer 5: convolution. Input 4x34x64. Output 1x16x64
    model.add(Convolution2D(64, 3, 3, border_mode='valid', name='conv5'))
    model.add(MaxPooling2D((2, 2), border_mode='valid', name='pool5'))
    model.add(Activation('relu', name='act5'))

    # flatten: Input 1x16x64. Output 1024 
    model.add(Flatten(name='flat'))

    # layer 6: fully connected + dropout. Input 1024. Output 100
    model.add(Dense(100, name='fc6'))
    model.add(Dropout(dropout[0], name='drop6'))
    model.add(Activation('relu', name='act6'))

    # layer 7: fully connected + dropout. Input 100. Output 50
    model.add(Dense(50, name='fc7'))
    model.add(Dropout(dropout[1], name='drop7'))
    model.add(Activation('relu', name='act7'))

    # layer 8: fully connected + dropout. Input 50. Output 10
    model.add(Dense(10, name='fc8'))
    model.add(Dropout(dropout[2], name='drop8'))
    model.add(Activation('relu', name='act8'))

    # layer 9: fully connected. Input 10. Output 1.
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

    save_best = ModelCheckpoint("{}.hdf5".format(exp_name), save_best_only=True, verbose=1)

    history_object = model.fit_generator(train_generator, samples_per_epoch=n_train,
            validation_data=valid_generator, nb_val_samples=n_valid, nb_epoch=nb_epoch,
            callbacks=[save_best])

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('results/loss_{}.png'.format(exp_name))

