# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import math
import numpy as np
import tarfile
from os import path
import logging

import cntk
from cntk.logging import *
from cntk import input as cntk_input 
from cntk import cross_entropy_with_softmax, classification_error
from cntk import Trainer, cntk_py 
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from cntk.debugging import set_computation_network_trace_level
from cntk.device import try_set_default_device, gpu
from cntk import data_parallel_distributed_learner, block_momentum_distributed_learner, Communicator
from cntk.train.training_session import *
from cntk.debugging import *
import cntk.io.transforms as xforms
from cntk.io import StreamDefs, StreamDef, ImageDeserializer, MinibatchSource
from resnet_models import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

# model dimensions - these match the ones from convnet_cifar10_dataaug
# so we can use the same data source
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10
model_name   = "ResNet_CIFAR10_DataAug.model"



def extract(fname):
    datapath = os.path.split(fname)[0]
    print ('Extracting files to {}...'.format(datapath))
    with tarfile.open(fname) as tar:
        tar.extractall(path=datapath)
    print ('Done.')


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


def process_data(fname):
    """ Uncompresses the files
    """
    if not path.isdir(path.join(path.split(fname)[0], 'train')):
        extract(fname)
    else:
        logger.info("Data already extracted. Skipping...")

        
def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    """ Creates minibatch source
    """
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

   
    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        imgfolder = os.path.join(os.path.split(map_file)[0], 'train')
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
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
            features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=train,
        epoch_size=total_number_of_samples,
        multithreaded_deserializer = True)


# Create network
def create_resnet_network(network_name):
    # Input variables denoting the features and label data
    input_var = cntk_input((num_channels, image_height, image_width))
    label_var = cntk_input((num_classes))

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
        'name' : network_name,
        'feature': input_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }



# Create trainer
def create_trainer(network, minibatch_size, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer):
    if network['name'] == 'resnet20': 
        lr_per_mb = [1.0]*80+[0.1]*40+[0.01]
    elif network['name'] == 'resnet110': 
        lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
    else: 
        return RuntimeError("Unknown model name!")

    momentum_time_constant = -minibatch_size/np.log(0.9)
    l2_reg_weight = 0.0001

    # Set learning parameters
    lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    
    # learner object
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    local_learner = momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule,
                                 l2_regularization_weight = l2_reg_weight)

    if block_size != None:
        learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)
    
    return Trainer(network['output'], (network['ce'], network['pe']), learner, progress_printer)


# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling=False):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    if profiling:
        start_profiler(sync_gpu=True)
        
    training_session(
        trainer=trainer, mb_source = train_source, 
        mb_size = minibatch_size,
        model_inputs_to_streams = input_map,
        checkpoint_config = CheckpointConfig(filename = os.path.join(model_path, model_name), restore=restore),
        progress_frequency=epoch_size,
        test_config = TestConfig(source=test_source, mb_size=16)
    ).train()
    
    if profiling:
        stop_profiler()


# Train and evaluate the network.
def resnet_cifar10(train_data, test_data, mean_data, network_name, epoch_size, num_quantization_bits=32, block_size=3200, warm_up=0, 
                   max_epochs=5, restore=True, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=False, scale_up=False, profiling=False):

    set_computation_network_trace_level(0)
    
    # NOTE: scaling up minibatch_size increases sample throughput. In 8-GPU machine,
    # ResNet110 samples-per-second is ~7x of single GPU, comparing to ~3x without scaling
    # up. However, bigger minimatch size on the same number of samples means less updates, 
    # thus leads to higher training error. This is a trade-off of speed and accuracy
    minibatch_size = 128 * (Communicator.num_workers() if scale_up else 1)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_resnet_network(network_name)
    trainer = create_trainer(network, minibatch_size, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer)
    train_source = create_image_mb_source(train_data, mean_data, train=True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, train=False, total_number_of_samples=cntk.io.FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, profiling)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='network type, resnet20 or resnet110', required=False, default='resnet20')
    parser.add_argument('-s', '--scale_up', help='scale up minibatch size with #workers for better parallelism', type=bool, required=False, default='False')
    parser.add_argument('-datafile', '--datafile', help='Data directory where the CIFAR dataset is located', required=True)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-e', '--epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-es', '--epoch_size', help='Size of epoch in samples', type=int, required=False, default=None)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)

    args = vars(parser.parse_args())

    epoch_size = 50000
    if args['outputdir'] != None:
        model_path = args['outputdir'] + "/models"

    if args['device'] != None:
        try_set_default_device(gpu(args['device']))

    if args['epoch_size'] is not None:
        epoch_size = args['epoch_size']

    data_file = args['datafile']

    if not os.path.exists(data_file):
        raise RuntimeError("File %s does not exist" % data_file)

    process_data(data_file) # uncompress the files create images
    data_path = os.path.split(data_file)[0]
    mean_data=os.path.join(data_path, 'CIFAR-10_mean.xml')
    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'test_map.txt')

    num_quantization_bits = args['quantized_bits']
    epochs = args['epochs']
    warm_up = args['distributed_after']
    network_name = args['network']
    scale_up = bool(args['scale_up'])

    # Create distributed trainer factory
    logger.info("Start training: quantize_bit = {}, epochs = {}, distributed_after = {}".format(num_quantization_bits, epochs, warm_up))

    try:
        resnet_cifar10(train_data, test_data, mean_data,
                       network_name, 
                       epoch_size,
                       num_quantization_bits,
                       block_size=args['block_samples'],
                       warm_up=args['distributed_after'],
                       max_epochs=epochs,
                       restore=not args['restart'],
                       scale_up=scale_up,
                       log_to_file=args['logdir'],
                       profiling=args['profile'])
    finally:
        # Must call MPI finalize when process exit
        Communicator.finalize()
