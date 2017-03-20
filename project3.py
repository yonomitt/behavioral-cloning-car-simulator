
### This script will allow me to run networks with different parameters on the fly,
### facilitating quick experiments.

import argparse
import glob
import sys

from model import *


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('ERROR: %s\n\n' % message)
        self.print_help()
        sys.exit(2)


if __name__ == '__main__':

    parser = ArgParser(description='Trains a model to drive a car simulator by example', add_help=False)
    dummy = parser.add_argument('-m', '--model', type=str, help='Name of the model to run', default='conv_3_fc_3')
    dummy = parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', default=5)
    dummy = parser.add_argument('-b', '--batch_size', type=int, help='Number of samples in a batch', default=32)
    dummy = parser.add_argument('-d', '--dropout', type=float, nargs='+', help='Dropout rates for the layers', default=[0.0])
    dummy = parser.add_argument('-s', '--sample_sets', type=str, nargs='+', help='Sample set(s) to use', default=None)
    dummy = parser.add_argument('-i', '--identifier', type=str, help='Extra identifier to add to saved files', default=None)
    dummy = parser.add_argument('-v', '--valid_pct', type=float, help='Percentage of the sample set to use for validation', default=0.2)
    dummy = parser.add_argument('-x', '--steering_correction', type=float, help='Steering correction if using left and right cameras', default=STEERING_CORRECTION)
    dummy = parser.add_argument('-z', '--zeros_to_ignore', type=float, help='Percentage of zero angle steering samples to remove', default=0.0)
    dummy = parser.add_argument('-c', '--center_only', action='store_true', help='Only use the center camera images')


    args = parser.parse_args()

    if not args.model in dir():
        print("Model '{}' NOT FOUND".format(args.model))
        sys.exit(-1)

    identifier = args.identifier

    gen_model = globals()[args.model]

    batch_size = args.batch_size
    nb_epoch = args.epochs
    valid_pct = args.valid_pct
    steering_correction = args.steering_correction
    zeros_to_ignore = args.zeros_to_ignore
    center_only = args.center_only


    dropout = args.dropout

    model = gen_model(dropout=dropout)
    model.summary()
    
    exp_parts = []

    if identifier:
        exp_parts.append(identifier)

    exp_parts.append(model.name)
    exp_parts.append('b{}'.format(batch_size))
    exp_parts.append('e{}'.format(nb_epoch))
    exp_parts.append('d{}'.format('_'.join('{:.02f}'.format(d) for d in dropout)))
    exp_parts.append('s{}'.format(steering_correction))
    exp_parts.append('v{}'.format(valid_pct))

    if zeros_to_ignore:
        exp_parts.append('z{}'.format(zeros_to_ignore))

    if center_only:
        exp_parts.append('center')

    exp_name = '-'.join(exp_parts)

    plot(model, show_shapes=True, to_file='results/{}-model.png'.format(exp_name))

    # get data
    sample_sets = args.sample_sets or glob.glob('data/*')
    samples = read_samples(sample_sets, center_only=center_only, zeros_to_ignore=zeros_to_ignore, steering_correction=steering_correction)

    nb_samples = len(samples)
    nb_valid = round(nb_samples * valid_pct)
    nb_train = nb_samples - nb_valid

    train_samples = samples[:nb_train]
    valid_samples = samples[nb_train:]

    resize = (model.input_shape[2], model.input_shape[1])

    train_generator = data_generator(train_samples, resize=resize, batch_size=batch_size)
    valid_generator = data_generator(valid_samples, resize=resize, batch_size=batch_size)

    model.compile(loss='mse', optimizer='adam')

    print("nb_samples: {}".format(nb_samples))
    print("nb_train: {}".format(nb_train))
    print("nb_valid: {}".format(nb_valid))

    save_best = ModelCheckpoint("results/{}.hdf5".format(exp_name), save_best_only=True, verbose=1)
    stop_early = EarlyStopping(patience=4, verbose=1)

    history_object = model.fit_generator(train_generator, samples_per_epoch=nb_train,
            validation_data=valid_generator, nb_val_samples=nb_valid, nb_epoch=nb_epoch,
            callbacks=[save_best, stop_early])

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('results/{}-loss.png'.format(exp_name))
