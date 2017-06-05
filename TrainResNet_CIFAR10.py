# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function

import argparse
import logging

import cntk
import cntk.io.transforms as xforms
import numpy as np
from cntk import Trainer
from cntk import cross_entropy_with_softmax, classification_error
from cntk.debugging import *
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from cntk.logging import *
from resnet_models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10
model_name   = "ResNet_CIFAR10_DataAug.model"


def process_map_file(map_file, imgfolder):
    """ Convert map file format to one required by CNTK ImageDeserializer
    """
    logger.info('Processing {}...'.format(map_file))
    orig_file = open(map_file, 'r')
    map_path, map_name = os.path.split(map_file)
    new_filename = os.path.join(map_path, 'p_{}'.format(map_name))
    new_file = open(new_filename, 'w')
    for line in orig_file:
        fname, label = line.split('\t')
        new_file.write("%s\t%s\n" % (os.path.join(imgfolder, fname), label.strip()))
    orig_file.close()
    new_file.close()
    return new_filename


def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    """ Creates minibatch source
    """
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError(
            "File '%s' or '%s' does not exist. " %
            (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        imgfolder = os.path.join(os.path.split(map_file)[0], 'train')
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio')  # train uses jitter
        ]
    else:
        imgfolder = os.path.join(os.path.split(map_file)[0], 'test')

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]

    map_file = process_map_file(map_file, imgfolder)

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features=StreamDef(field='image', transforms=transforms),
            # first column in map file is referred to as 'image'
            labels=StreamDef(field='label', shape=num_classes))),  # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)


#
# # Train and evaluate the network.
# def train_and_evaluate(reader_train, reader_test, network_name, epoch_size, max_epochs,
#                        model_dir=None, tensorboard_logdir=None):
#
#     set_computation_network_trace_level(0)
#
#     # Input variables denoting the features and label data
#     input_var = C.input_variable((num_channels, image_height, image_width))
#     label_var = C.input_variable((num_classes))
#
#     # create model, and configure learning parameters
#     if network_name == 'resnet20':
#         z = create_cifar10_model(input_var, 3, num_classes)
#         lr_per_mb = [1.0]*80+[0.1]*40+[0.01]
#     elif network_name == 'resnet110':
#         z = create_cifar10_model(input_var, 18, num_classes)
#         lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
#     else:
#         return RuntimeError("Unknown model name!")
#
#     # loss and metric
#     ce = cross_entropy_with_softmax(z, label_var)
#     pe = classification_error(z, label_var)
#
#     # shared training parameters
#     minibatch_size = 128
#     momentum_time_constant = -minibatch_size/np.log(0.9)
#     l2_reg_weight = 0.0001
#
#     # Set learning parameters
#     lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
#     lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
#     mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
#
#     # progress writers
#     progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs)]
#     tensorboard_writer = None
#     if tensorboard_logdir is not None:
#         tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z)
#         progress_writers.append(tensorboard_writer)
#
#     # trainer object
#     learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule,
#                            l2_regularization_weight = l2_reg_weight)
#     trainer = Trainer(z, (ce, pe), learner, progress_writers)
#
#     # define mapping from reader streams to network inputs
#     input_map = {
#         input_var: reader_train.streams.features,
#         label_var: reader_train.streams.labels
#     }
#
#     log_number_of_parameters(z) ; print()
#
#     # perform model training
#     if profiler_dir:
#         start_profiler(profiler_dir, True)
#
#     for epoch in range(max_epochs):       # loop over epochs
#         sample_count = 0
#         while sample_count < epoch_size:  # loop over minibatches in the epoch
#             data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
#             trainer.train_minibatch(data)                                   # update model with it
#             sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
#
#         trainer.summarize_training_progress()
#
#         # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
#         if tensorboard_writer:
#             for parameter in z.parameters:
#                 tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)
#
#         if model_dir:
#             z.save(os.path.join(model_dir, network_name + "_{}.dnn".format(epoch)))
#         enable_profiler() # begin to collect profiler data after first epoch
#
#     if profiler_dir:
#         stop_profiler()
#
#     # Evaluation parameters
#     test_epoch_size     = 10000
#     minibatch_size = 16
#
#     # process minibatches and evaluate the model
#     metric_numer    = 0
#     metric_denom    = 0
#     sample_count    = 0
#
#     while sample_count < test_epoch_size:
#         current_minibatch = min(minibatch_size, test_epoch_size - sample_count)
#         # Fetch next test min batch.
#         data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
#         # minibatch data to be trained with
#         metric_numer += trainer.test_minibatch(data) * current_minibatch
#         metric_denom += current_minibatch
#         # Keep track of the number of samples processed so far.
#         sample_count += data[label_var].num_samples
#
#     print("")
#     trainer.summarize_test_progress()
#     print("")
#
#     return metric_numer/metric_denom


# Create network
def create_resnet_network(network_name):
    # Input variables denoting the features and label data
    input_var = cntk.input_variable((num_channels, image_height, image_width))
    label_var = cntk.input_variable((num_classes))

    # create model, and configure learning parameters
    if network_name == 'resnet20':
        z = create_cifar10_model(input_var, 3, num_classes)
    elif network_name == 'resnet110':
        z = create_cifar10_model(input_var, 18, num_classes)
    else:
        return RuntimeError("Unknown model name!")

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    return {
        'name': network_name,
        'feature': input_var,
        'label': label_var,
        'ce': ce,
        'pe': pe,
        'output': z
    }


# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore):
    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    cntk.training_session(
        trainer=trainer,
        mb_source=train_source,
        mb_size=minibatch_size,
        model_inputs_to_streams=input_map,
        checkpoint_config=cntk.CheckpointConfig(filename=os.path.join(model_path, model_name), restore=restore),
        progress_frequency=epoch_size,
        test_config=cntk.TestConfig(source=test_source, mb_size=16)
    ).train()



# Create trainer
def create_trainer(network, minibatch_size, epoch_size, progress_printer):
    if network['name'] == 'resnet20':
        lr_per_mb = [1.0] * 80 + [0.1] * 40 + [0.01]
    elif network['name'] == 'resnet110':
        lr_per_mb = [0.1] * 1 + [1.0] * 80 + [0.1] * 40 + [0.01]
    else:
        return RuntimeError("Unknown model name!")

    momentum_time_constant = -minibatch_size / np.log(0.9)
    l2_reg_weight = 0.0001

    # Set learning parameters
    lr_per_sample = [lr / minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    learner = momentum_sgd(network['output'].parameters,
                           lr_schedule,
                           mm_schedule,
                           l2_regularization_weight=l2_reg_weight)

    return Trainer(network['output'], (network['ce'], network['pe']), learner, progress_printer)


# Train and evaluate the network.
def resnet_cifar10(train_source,
                   test_source,
                   network_name,
                   epoch_size,
                   minibatch_size = 128,
                   max_epochs=5,
                   log_file=None,
                   tboard_log_dir='.'):

    set_computation_network_trace_level(0)

    network = create_resnet_network(network_name)

    progress_printer = ProgressPrinter(
        tag='Training',
        log_to_file=log_file,
        rank=cntk.Communicator.rank(),
        num_epochs=max_epochs)

    tensorboard_writer = TensorBoardProgressWriter(freq=10,
                                                   log_dir=tboard_log_dir,
                                                   model=network['output'])
    trainer = create_trainer(network, minibatch_size, epoch_size, [progress_printer, tensorboard_writer])
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='network type, resnet20 or resnet110', required=False,
                        default='resnet20')
    parser.add_argument('--datafolder', help='Data directory where the CIFAR dataset is located',
                        required=True)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False,
                        default=None)
    parser.add_argument('-logfile', '--logfile', help='Log file', required=False, default=None)
    parser.add_argument('-e', '--epochs', help='Total number of epochs to train', type=int, required=False,
                        default='160')
    parser.add_argument('-es', '--epoch_size', help='Size of epoch in samples', type=int, required=False, default=None)
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
    parser.add_argument('-m', '--modeldir', help='directory for saving model', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where TensorBoard logs should be created', required=False, default=None)

    args = vars(parser.parse_args())
    epochs = int(args['epochs'])
    network_name = args['network']

    model_dir = args['modeldir']
    if not model_dir:
        model_dir = os.path.join(abs_path, "Models")

    data_path = args['datafolder']
    if not os.path.exists(data_path):
        raise RuntimeError("Folder %s does not exist" % data_path)

    epoch_size = 50000

    train_source = create_image_mb_source(os.path.join(data_path, 'train_map.txt'),
                                          os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                          train=True,
                                          total_number_of_samples=epochs * epoch_size)

    test_source = create_image_mb_source(os.path.join(data_path, 'test_map.txt'),
                                         os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                         train=False,
                                         total_number_of_samples=cntk.io.FULL_DATA_SWEEP)

    resnet_cifar10(train_source,
                   test_source,
                   network_name,
                   epoch_size,
                   max_epochs=epochs,
                   log_file=args['logfile'],
                   tboard_log_dir=args['profile'])
