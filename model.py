from sys import platform

if platform == 'linux':
    import matplotlib
    matplotlib.use('Agg')

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

import keras

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


if keras.__version__.startswith('1'):
    from keras.utils.visualize_util import plot
else:
    from keras.utils.vis_utils import plot_model as plot


# steering correction in degrees
STEERING_CORRECTION = 0.1
VALIDATION_PCT = 0.2


def conv_4_fc_3_more_filters(dropout = []):

    """This network has four convolution layers and three fully connected layers and a lot more
    filters than the conv_4_fc_3 model.

    Parameters:
        dropout - list of dropout values for the 3 fully connected layers
        
    Returns:
        A model"""

    # ensure the dropout list has enough values in it for the model
    # augment the values of some are missing
    if dropout == None or len(dropout) == 0:
        dropout = [0.0, 0.0]
    elif len(dropout) == 1:
        dropout = dropout * 2

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

    # crop top 56 rows and bottom 24 rows from the images
    model.add(Cropping2D(cropping=((56, 24), (0, 0)), input_shape=(160, 320, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # layer 1: convolution + max pooling. Input 80x320x3. Output 40x160x32
    model.add(Convolution2D(32, 5, 5, border_mode='same', name='conv1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Activation('relu', name='act1'))

    # layer 2: convolution = max pooling. Input 40x160x32. Output 20x80x64
    model.add(Convolution2D(64, 5, 5, border_mode='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    model.add(Activation('relu', name='act2'))

    # layer 3: convolution = max pooling. Input 20x80x64. Output 10x40x128
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv3'))
    model.add(MaxPooling2D((2, 2), name='pool3'))
    model.add(Activation('relu', name='act3'))

    # layer 3: convolution = max pooling. Input 10x40x128. Output 5x20x128
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv4'))
    model.add(MaxPooling2D((2, 2), name='pool4'))
    model.add(Activation('relu', name='act4'))

    # flatten: Input 5x20x128. Output 12800
    model.add(Flatten(name='flat'))

    # layer 4: fully connected + dropout. Input 12800. Output 556
    model.add(Dense(556, name='fc5'))
    model.add(Dropout(dropout[0], name='drop5'))
    model.add(Activation('relu', name='act5'))

    # layer 5: fully connected + dropout. Input 556. Output 24
    model.add(Dense(24, name='fc6'))
    model.add(Dropout(dropout[1], name='drop6'))
    model.add(Activation('relu', name='act6'))

    # layer 6: fully connected. Input 24. Output 1.
    model.add(Dense(1, name='out'))

    return model


def conv_4_fc_3(dropout = []):

    """This network has four convolution layers and three fully connected layers

    Parameters:
        dropout - list of dropout values for the 3 fully connected layers
        
    Returns:
        A model"""

    # ensure the dropout list has enough values in it for the model
    # augment the values of some are missing
    if dropout == None or len(dropout) == 0:
        dropout = [0.0, 0.0]
    elif len(dropout) == 1:
        dropout = dropout * 2

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

    # crop top 56 rows and bottom 24 rows from the images
    model.add(Cropping2D(cropping=((56, 24), (0, 0)), input_shape=(160, 320, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # layer 1: convolution + max pooling. Input 80x320x3. Output 40x160x8
    model.add(Convolution2D(8, 5, 5, border_mode='same', name='conv1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Activation('relu', name='act1'))

    # layer 2: convolution = max pooling. Input 40x160x8. Output 20x80x16
    model.add(Convolution2D(16, 5, 5, border_mode='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    model.add(Activation('relu', name='act2'))

    # layer 3: convolution = max pooling. Input 20x80x16. Output 10x40x32
    model.add(Convolution2D(32, 3, 3, border_mode='same', name='conv3'))
    model.add(MaxPooling2D((2, 2), name='pool3'))
    model.add(Activation('relu', name='act3'))

    # layer 3: convolution = max pooling. Input 10x40x32. Output 5x20x32
    model.add(Convolution2D(32, 3, 3, border_mode='same', name='conv4'))
    model.add(MaxPooling2D((2, 2), name='pool4'))
    model.add(Activation('relu', name='act4'))

    # flatten: Input 5x20x32. Output 3200
    model.add(Flatten(name='flat'))

    # layer 4: fully connected + dropout. Input 3200. Output 556
    model.add(Dense(556, name='fc5'))
    model.add(Dropout(dropout[0], name='drop5'))
    model.add(Activation('relu', name='act5'))

    # layer 5: fully connected + dropout. Input 556. Output 24
    model.add(Dense(24, name='fc6'))
    model.add(Dropout(dropout[1], name='drop6'))
    model.add(Activation('relu', name='act6'))

    # layer 6: fully connected. Input 24. Output 1.
    model.add(Dense(1, name='out'))

    return model


def resnet_ish(dropout = []):

    """This model attempts to use transfer learning on ResNet50

    Parameters:
        dropout - list of dropout values for the 2 fully connected layers
        
    Returns:
        A model with ResNet50 at it's core"""
        
    # ensure the dropout list has enough values in it for the model
    # augment the values of some are missing
    if dropout == None or len(dropout) == 0:
        dropout = [0.0, 0.0, 0.0]
    elif len(dropout) == 1:
        dropout = dropout * 3
    elif len(dropout) == 2:
        dropout.append(dropout[1])

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

    # crop top 157 rows and bottom 67 rows from the images
    model.add(Cropping2D(cropping=((157, 67), (0, 0)), input_shape=(448, 224, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # load the ResNet50 model with weights from imagenet. Do not include the top of the model
    # as it will be replaced and trained for driving in the simulator.
    resnet = ResNet50(weights='imagenet', include_top=False)

    # freeze the ResNet50 layers to speed up training
    for layer in resnet.layers:
        layer.trainable = False

    model.add(resnet)

    # flatten
    model.add(Flatten())

    # layer 153. fully connected + dropout. Input 2048. Output 1000.
    model.add(Dense(1000, name='fc153'))
    model.add(Dropout(dropout[0], name='drop153'))
    model.add(Activation('relu', name='act153'))

    # layer 154. fully connected + dropout. Input 1000. Output 100.
    model.add(Dense(100, name='fc154'))
    model.add(Dropout(dropout[1], name='drop154'))
    model.add(Activation('relu', name='act154'))

    # layer 155: fully connected. Input 100. Output 1.
    model.add(Dense(1, name='out'))

    return model


def vgg16_ish(dropout = []):

    """This model attempts to use transfer learning on VGG16

    Parameters:
        dropout - list of dropout values for the 2 fully connected layers
        
    Returns:
        A model with VGG16 at it's core"""
        
    # ensure the dropout list has enough values in it for the model
    # augment the values of some are missing
    if dropout == None or len(dropout) == 0:
        dropout = [0.0, 0.0, 0.0]
    elif len(dropout) == 1:
        dropout = dropout * 3
    elif len(dropout) == 2:
        dropout.append(dropout[1])

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

    # crop top 157 rows and bottom 67 rows from the images
    model.add(Cropping2D(cropping=((157, 67), (0, 0)), input_shape=(448, 224, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # load the VGG16 model with weights from imagenet. Do not include the top of the model
    # as it will be replaced and trained for driving in the simulator.
    vgg16 = VGG16(weights='imagenet', include_top=False)

    # freeze the ResNet50 layers to speed up training
    for layer in vgg16.layers:
        layer.trainable = False

    model.add(vgg16)

    # flatten
    model.add(Flatten())

    # layer 17. fully connected + dropout. Input 25088. Output 1000.
    model.add(Dense(1000, name='fc17'))
    model.add(Dropout(dropout[0], name='drop17'))
    model.add(Activation('relu', name='act17'))

    # layer 18. fully connected + dropout. Input 1000. Output 100.
    model.add(Dense(100, name='fc18'))
    model.add(Dropout(dropout[1], name='drop18'))
    model.add(Activation('relu', name='act18'))

    # layer 19: fully connected. Input 100. Output 1.
    model.add(Dense(1, name='out'))

    return model


def end_to_end_nvidia(dropout = []):

    """This model attempts to mimic the model by NVIDIA in their paper End to End Learning for Self-Driving
    Cars:

    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    Parameters:
        dropout - list of dropout values for the 3 fully connected layers
        
    Returns:
        A model based on the End to End NVIDIA model"""
        
    # ensure the dropout list has enough values in it for the model
    # augment the values of some are missing
    if dropout == None or len(dropout) == 0:
        dropout = [0.0, 0.0, 0.0]
    elif len(dropout) == 1:
        dropout = dropout * 3
    elif len(dropout) == 2:
        dropout.append(dropout[1])

    # this hack gets the current function name and sets it to the name of the model
    model = Sequential(name=traceback.extract_stack(None, 2)[-1][2])

    # crop top 56 rows and bottom 24 rows from the images
    model.add(Cropping2D(cropping=((56, 24), (0, 0)), input_shape=(160, 320, 3), name='pp_crop'))

    # mean center the pixels
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='pp_center'))

    # layer 1: convolution. Input 40x160x3. Output 36x156x24
    model.add(Convolution2D(24, 5, 5, border_mode='valid', name='conv1'))
    model.add(MaxPooling2D((2, 2), border_mode='valid', name='pool1'))
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

    
def data_generator(samples, resize=None, batch_size=128):

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
                if resize:
                    image = cv2.resize(image, resize)
                if batch_sample[2]:
                    image = cv2.flip(image, 1)
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


def read_samples(base_dirs, center_only=False, zeros_to_ignore=0.0, steering_correction=STEERING_CORRECTION):

    """Read the samples from CSV files

    Parameters:
        - base_dirs: list of directories containing the CSV files
        
    Returns: list of tuples containing the absolute path to the image and the normalized steering angle"""

    samples = []

    def dont_ignore(steering, percent):
        return (steering != 0.0) or (np.random.random() < (1 - percent))

    for base_dir in base_dirs:

        with open(os.path.join(base_dir, "driving_log.csv")) as f:
            log = [l.split(',') for l in f.read().split('\n')[1:-1]]

            for l in log:
                center_steering = float(l[3])# / 25.0
                left_steering = (float(l[3]) + steering_correction)# / 25.0
                right_steering = (float(l[3]) - steering_correction)# / 25.0

                # center image
                if dont_ignore(center_steering, zeros_to_ignore):
                    samples.append((gen_rel_img(base_dir, l[0]), center_steering, False))

                if not center_only:
                    # left image
                    if dont_ignore(left_steering, zeros_to_ignore):
                        samples.append((gen_rel_img(base_dir, l[1]), left_steering, False))
                    # right image
                    if dont_ignore(right_steering, zeros_to_ignore):
                        samples.append((gen_rel_img(base_dir, l[2]), right_steering, False))

                # mirror images
                if center_steering != 0.0:
                    samples.append((gen_rel_img(base_dir, l[0]), -center_steering, True))

                if not center_only:
                    if left_steering != 0.0:
                        samples.append((gen_rel_img(base_dir, l[1]), -left_steering, True))
    
                    if right_steering != 0.0:
                        samples.append((gen_rel_img(base_dir, l[2]), -right_steering, True))

    return samples


if __name__ == '__main__':

    batch_size = 32
    nb_epoch = 5

    model = conv_4_fc_3(dropout=[0.2, 0.5])
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

    resize = (model.input_shape[2], model.input_shape[1])

    train_generator = data_generator(train_samples, resize=resize, batch_size=batch_size)
    valid_generator = data_generator(valid_samples, resize=resize, batch_size=batch_size)

    model.compile(loss='mse', optimizer='adam')

    print("n_samples: {}".format(n_samples))
    print("n_train: {}".format(n_train))
    print("n_valid: {}".format(n_valid))

    save_best = ModelCheckpoint("{}.hdf5".format(exp_name), save_best_only=True, verbose=1)
    stop_early = EarlyStopping(patience=4, verbose=1)

    history_object = model.fit_generator(train_generator, samples_per_epoch=n_train,
            validation_data=valid_generator, nb_val_samples=n_valid, nb_epoch=nb_epoch,
            callbacks=[save_best, stop_early])

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('results/loss_{}.png'.format(exp_name))

