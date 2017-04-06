#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
# from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          # Permute)
from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import *
from deeprl_hw2.utils import *
from deeprl_hw2.policy import *
from deeprl_hw2.core import *
import gym
from deeprl_hw2.constants import *


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    if model_name == 0:
        model = linear_model(window, input_shape, num_actions)
    elif model_name == 1:
        model = deep_model(window, input_shape, num_actions)
    elif model_name == 2:
        model = dueling_deep(window, input_shape, num_actions)
    else:
        print("No suitable models found.")
        exit()
    print(model.summary())
    return model

def linear_model(window, input_shape, num_actions):
    inputs = Input(shape=(input_shape) + (window,), name='input')
    flattened_input = Flatten()(inputs)
    with tf.name_scope('output'):
        output = Dense(num_actions, activation='linear')(flattened_input)
    model = Model(inputs=inputs, outputs=output, name='linear_q_network')
    return model


def deep_model(window, input_shape, num_actions):
    inputs = Input(shape=(input_shape) + (window,), name='input')
    with tf.name_scope('hidden1'):
        hidden1 = Conv2D(16, (8, 8), strides=4, activation='relu')(inputs)
    with tf.name_scope('hidden2'):
        hidden2 = Conv2D(32, (4, 4), strides=2, activation='relu')(hidden1)
    flattened = Flatten()(hidden2)
    with tf.name_scope('hidden3'):
        hidden3 = Dense(256, activation='relu')(flattened)
    with tf.name_scope('output'):
        output = Dense(num_actions)(hidden3)
    return Model(inputs = inputs, outputs = output, name='deep_model')

def dueling_deep(window, input_shape, num_actions):
    inputs = Input(shape=(input_shape) + (window,), name='input')
    with tf.name_scope('hidden1'):
        hidden1 = Conv2D(16, (8, 8), strides=4, activation='relu')(inputs)
    with tf.name_scope('hidden2'):
        hidden2 = Conv2D(32, (4, 4), strides=2, activation='relu')(hidden1)
    flattened = Flatten()(hidden2)
    with tf.name_scope('stream1fully'):
        stream1 = Dense(256, activation='relu')(flattened)
    with tf.name_scope('stream2fully'):
        stream2 = Dense(256, activation='relu')(flattened)
    with tf.name_scope('stream1value'):
        V = Dense(1)(stream1)
    with tf.name_scope('stream2advantage'):
        As = Dense(num_actions)(stream2)
    from deeprl_hw2.dqn import MyLayer
    with tf.name_scope('output'):
        Q = MyLayer()([V,As])
    #Q = V + As - K.mean(As, axis=1, keepdims=True)
    return Model(inputs=inputs, outputs=Q, name='dueling_deep')

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--model', default=1, type=int, help='model')
    parser.add_argument('--double', action='store_true')

    args = parser.parse_args()
    print('Using Tensorflow Version of ' + tf.__version__)
    # args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
    print("Output Folder: " + args.output)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    sess = tf.Session()
    K.set_session(sess)


    env = gym.make(args.env)
    num_actions = env.action_space.n
    # 0 linear; 1 deep; 2 dueling
    model = create_model(WINDOW, INPUT_SHAPE, num_actions, args.model)
    atari_preprocessor = AtariPreprocessor(INPUT_SHAPE)
    history_preprocessor = HistoryPreprocessor(HIST_LENGTH)
    preprocessor = PreprocessorSequence([atari_preprocessor, history_preprocessor])
    memory = ReplayMemory(MAX_MEMORY, WINDOW)
    policy = LinearDecayGreedyEpsilonPolicy(START_EPSILON, END_EPSILON, NUM_STEPS)

    dqn_agent = DQNAgent(model, num_actions, preprocessor, memory, policy, GAMMA,
            TARGET_UPDATE_FREQ, INIT_MEMORY, TRAIN_FREQ, BATCH_SIZE, double=args.double)

    optimizer = Adam(lr=LEARNING_RATE, epsilon=MIN_SQ_GRAD)
    loss_func = mean_huber_loss
    dqn_agent.compile(optimizer, loss_func)
    dqn_agent.fit(env, NUM_ITERATIONS,  MAX_EPISODE_LENGTH)


if __name__ == '__main__':
    main()
