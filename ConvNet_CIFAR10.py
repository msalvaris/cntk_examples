# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
from cntk.ops import minus, element_times, constant, relu
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk import learners, io, logging, layers, Trainer, learning_rate_schedule, reduce_mean
import _cntk_py
import cntk
import fire
from uuid import uuid4
from toolz import pipe
import json


# Paths relative to current python file.
abs_path   = os.getcwd()
data_path  = abs_path
model_path = os.path.join(abs_path, "Models")


def _create_env_variable_appender(env_var_name):
    def env_var_appender(identifier):
        env_var_value = os.environ.get(env_var_name, None)
        if env_var_value is None:
            return identifier
        else:
            return '{}_{}'.format(identifier, env_var_value)
    return env_var_appender


_append_task_id = _create_env_variable_appender('AZ_BATCH_TASK_ID') # Append task id if the env variable exists
_append_job_id = _create_env_variable_appender('AZ_BATCH_JOB_ID')   # Append job id if the env variable exists


def _get_unique_id():
    """ Returns a unique identifier

    If executed in a batch environment it will incorporate the job and task id
    """
    return pipe(str(uuid4())[:8],
                _append_task_id,
                _append_job_id)


def _save_results(test_result, filename, **kwargs):
    results_dict = {'test_metric':test_result, 'parameters': kwargs}
    with open(filename, 'w') as outfile:
        json.dump(results_dict, outfile)


# Define the reader for both training and evaluation action.
def create_reader(path, is_training, input_dim, label_dim):
    return io.MinibatchSource(io.CTFDeserializer(path, io.StreamDefs(
        features  = io.StreamDef(field='features', shape=input_dim),
        labels    = io.StreamDef(field='labels',   shape=label_dim)
    )), randomize=is_training, max_sweeps = io.INFINITELY_REPEAT if is_training else 1)


# Creates and trains a feedforward classification model for MNIST images
def convnet_cifar10(num_convolution_layers=2, minibatch_size=64, max_epochs=30, logdir=None, debug_output=False):
    _cntk_py.set_computation_network_trace_level(0)

    print("""Running network with: 
                {num_convolution_layers} convolution layers
                {minibatch_size}  minibatch size
                for {max_epochs} epochs""".format(
                    num_convolution_layers=num_convolution_layers,
                    minibatch_size=minibatch_size,
                    max_epochs=max_epochs
                ))
    
    image_height = 32
    image_width  = 32
    num_channels = 3
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    print('Setting up input variables')
    # Input variables denoting the features and label data
    input_var = cntk.ops.input((num_channels, image_height, image_width), np.float32)
    label_var = cntk.ops.input(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    input_removemean = minus(input_var, constant(128))
    scaled_input = element_times(constant(0.00390625), input_removemean)

    print('Creating NN model')
    with layers.default_options(activation=relu, pad=True): 
        model = layers.Sequential([
            layers.For(range(num_convolution_layers), lambda : [
                layers.Convolution2D((3,3), 64), 
                layers.Convolution2D((3,3), 64), 
                layers.MaxPooling((3,3), (2,2))
            ]), 
            layers.For(range(2), lambda i: [
                layers.Dense([256,128][i]), 
                layers.Dropout(0.5)
            ]), 
            layers.Dense(num_output_classes, activation=None)
        ])(scaled_input)
    
    ce = cross_entropy_with_softmax(model, label_var)
    pe = classification_error(model, label_var)

    reader_train = create_reader(os.path.join(data_path, 'Train_cntk_text.txt'), True, input_dim, num_output_classes)

    # training config
    epoch_size = 50000                  # for now we manually specify epoch size
    
    # Set learning parameters
    lr_per_sample          = [0.0015625]*10 + [0.00046875]*10 + [0.00015625]
    lr_schedule            = learning_rate_schedule(lr_per_sample, learners.UnitType.sample, epoch_size)
    mm_time_constant       = [0]*20 + [-minibatch_size/np.log(0.9)]
    mm_schedule            = learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size)
    l2_reg_weight          = 0.002

    # Instantiate the trainer object to drive the model training
    learner = learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule,
                                        l2_regularization_weight = l2_reg_weight)
     # progress writers
    progress_printer = [logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)]
    tensorboard_writer=None
    if logdir is not None:
        tensorboard_writer = logging.TensorBoardProgressWriter(freq=10, log_dir=logdir, model=model)
        progress_printer.append(tensorboard_writer)

    trainer = Trainer(model, (ce, pe), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var  : reader_train.streams.features,
        label_var  : reader_train.streams.labels
    }

    logging.log_number_of_parameters(model)

    print('Starting training')
    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far

        trainer.summarize_training_progress()
        # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
        if tensorboard_writer:
            for parameter in model.parameters:
                tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)
    
    # Load test data
    reader_test = create_reader(os.path.join(data_path, 'Test_cntk_text.txt'), False, input_dim, num_output_classes)

    input_map = {
        input_var  : reader_test.streams.features,
        label_var  : reader_test.streams.labels
    }

    # Test data for trained model
    epoch_size = 10000
    test_minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    print('Starting testing')
    while sample_count < epoch_size:
        current_minibatch = min(test_minibatch_size, epoch_size - sample_count)
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    # Save model and results
    unique_path = os.path.join(model_path, _get_unique_id())
    model.save(os.path.join(unique_path, "ConvNet_CIFAR10_model.dnn"))
    _save_results((metric_numer*100.0)/metric_denom,
                  os.path.join(unique_path, "model_results.json"),
                  num_convolution_layers=num_convolution_layers, 
                  minibatch_size=minibatch_size, 
                  max_epochs=max_epochs)

    
if __name__=='__main__':
    fire.Fire(convnet_cifar10)

