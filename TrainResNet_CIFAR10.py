# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function

import argparse
import logging
from functools import partial

import cntk
import cntk.io.transforms as xforms
import numpy as np
from cntk import Trainer, block_momentum_distributed_learner, data_parallel_distributed_learner
from cntk import cross_entropy_with_softmax, classification_error
from cntk.debugging import *
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from cntk.logging import *
from resnet_models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to current python file.
_ABS_PATH   = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_ABS_PATH, "Models")

# model dimensions
_IMAGE_HEIGHT = 32
_IMAGE_WIDTH  = 32
_NUM_CHANNELS = 3  # RGB
_NUM_CLASSES  = 10
_MODEL_NAME   = "ResNet_CIFAR10_DataAug.model"
_EPOCH_SIZE = 50000


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
        xforms.scale(width=_IMAGE_WIDTH, height=_IMAGE_HEIGHT, channels=_NUM_CHANNELS, interpolations='linear'),
        xforms.mean(mean_file)
    ]

    map_file = process_map_file(map_file, imgfolder)

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features=StreamDef(field='image', transforms=transforms),
            # first column in map file is referred to as 'image'
            labels=StreamDef(field='label', shape=_NUM_CLASSES))),  # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)



def create_resnet_network(network_name):
    """ Create network
    
    """
    # Input variables denoting the features and label data
    input_var = cntk.input_variable((_NUM_CHANNELS, _IMAGE_HEIGHT, _IMAGE_WIDTH))
    label_var = cntk.input_variable((_NUM_CLASSES))

    # create model, and configure learning parameters
    if network_name == 'resnet20':
        z = create_cifar10_model(input_var, 3, _NUM_CLASSES)
    elif network_name == 'resnet110':
        z = create_cifar10_model(input_var, 18, _NUM_CLASSES)
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



def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, model_path=_MODEL_PATH):
    """ Train and test
    
    """
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
        checkpoint_config=cntk.CheckpointConfig(filename=os.path.join(model_path, _MODEL_NAME), restore=restore),
        progress_frequency=epoch_size,
        test_config=cntk.TestConfig(source=test_source, mb_size=16)
    ).train()


def create_trainer(network, minibatch_size, epoch_size, progress_printer):
    """ Create trainer 
    """
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


def create_distributed_trainer(network, minibatch_size, epoch_size, progress_printer,
                               num_quantization_bits=32, block_size=0, warm_up=0):
    """ Create distributed trainer
    
    """
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

    # learner object
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    local_learner = momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule,
                                 l2_regularization_weight=l2_reg_weight)

    if block_size != None:
        learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits,
                                                    distributed_after=warm_up)

    return Trainer(network['output'], (network['ce'], network['pe']), learner, progress_printer)


def resnet_cifar10(train_source,
                   test_source,
                   network,
                   trainer_func,
                   epoch_size,
                   minibatch_size = 128,
                   max_epochs=5,
                   log_file=None,
                   tboard_log_dir='.'):
    """ Train and evaluate the network.
    """

    set_computation_network_trace_level(0)

    progress_printer = ProgressPrinter(
        tag='Training',
        log_to_file=log_file,
        rank=cntk.Communicator.rank(),
        num_epochs=max_epochs)

    tensorboard_writer = TensorBoardProgressWriter(freq=10,
                                                   log_dir=tboard_log_dir,
                                                   model=network['output'])
    trainer = trainer_func(network, minibatch_size, epoch_size, [progress_printer, tensorboard_writer])
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore=False)


def prepare_trainer(cmd_args):
    if args['distributed']:
        return partial(create_distributed_trainer,
                       num_quantization_bits = args['quantized_bits'],
                       block_size = args['block_samples'],
                       warm_up = args['distributed_after'])
    else:
        return create_trainer


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network',
                        help='network type, resnet20 or resnet110',
                        required=False,
                        default='resnet20')
    parser.add_argument('--datafolder',
                        help='Data directory where the CIFAR dataset is located',
                        required=True)
    parser.add_argument('-m', '--modeldir',
                        help='directory for saving model',
                        required=False,
                        default=None)
    parser.add_argument('-logfile', '--logfile', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created',
                        required=False,
                        default='.')
    parser.add_argument('-e', '--epochs',
                        help='Total number of epochs to train',
                        type=int,
                        required=False,
                        default='160')
    parser.add_argument('--distributed', help='Whether to run in distributed mode', required=False, default=False)
    parser.add_argument('-q', '--quantized_bits',
                        help='Number of quantized bits used for gradient aggregation',
                        type=int,
                        required=False,
                        default='32')
    parser.add_argument('-b', '--block_samples',
                        type=int,
                        help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)",
                        required=False,
                        default=None)
    parser.add_argument('-a', '--distributed_after',
                        help='Number of samples to train with before running distributed',
                        type=int,
                        required=False,
                        default='0')

    args = vars(parser.parse_args())
    epochs = int(args['epochs'])
    network_name = args['network']

    model_dir = args['modeldir']
    if not model_dir:
        model_dir = os.path.join(_ABS_PATH, "Models")

    data_path = args['datafolder']
    if not os.path.exists(data_path):
        raise RuntimeError("Folder %s does not exist" % data_path)

    train_source = create_image_mb_source(os.path.join(data_path, 'train_map.txt'),
                                          os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                          train=True,
                                          total_number_of_samples=epochs * _EPOCH_SIZE)

    test_source = create_image_mb_source(os.path.join(data_path, 'test_map.txt'),
                                         os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                         train=False,
                                         total_number_of_samples=cntk.io.FULL_DATA_SWEEP)

    network = create_resnet_network(network_name)

    resnet_cifar10(train_source,
                   test_source,
                   network,
                   prepare_trainer(args),
                   _EPOCH_SIZE,
                   max_epochs=epochs,
                   log_file=args['logfile'],
                   tboard_log_dir=args['tensorboard_logdir'])
