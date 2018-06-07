from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import matplotlib.pyplot as plt

import re
import os
import shutil
import argparse
import time
import gzip

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


from aux.dataset import Dataset
from aux.utils import check_folder


'''
This file contains methods to handle parsing
'''


def parse_args():

    desc = "Tensorflow implementation of VAE and GMVAE"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=20,  help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of inputs used for each iteration')
    # reconstrucion variance
    parser.add_argument('--sigma', type=float, default=0.1, help='Parameter that defines the variance of the output Gaussian distribution')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Parameter of the optimization function')

    parser.add_argument('--z_dim', type=int, default=3, help='Dimension of the latent variable z')
    parser.add_argument('--w_dim', type=int, default=3, help='Dimension of the latent variable w. Only for GMVAE')
    parser.add_argument('--K_clusters', type=int, default=10, help='Number of modes of the latent variable z. Only for GMVAE')

    # hidden units in the NNs
    parser.add_argument('--hidden_dim', type=int, default=50, help='Number of neurons of each dense layer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of dense layers in each network')

    parser.add_argument('--checkpoint_dir', type=str, default='saved_models', help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logdir', help='Directory name to save training logs')
    parser.add_argument('--data_dir', type=str, default='./MNIST_data', help='Directory for storing input data')

    parser.add_argument('--train', type=int, default=1, help='Flag to set train')
    parser.add_argument('--summary', type=int, default=1,help='Flag to set TensorBoard summary')
    parser.add_argument('--restore', type=int, default=0, help='Flag to restore model')
    parser.add_argument('--plot', type=int, default=0, help='Flag to set plot')
    parser.add_argument('--generate', type=int, default=0, help='Flag to set generation')

    parser.add_argument('--save_step', type=int, default=10, help='Save network each X epochs')


    parser.add_argument('--num_imgs', type=int, default=50,help='Images to plot')
    parser.add_argument('--dataset_name', type=str, default='MNIST or FREY')
    parser.add_argument('--model_name', type=str, default='model1', help='Fixes the model and architecture')
    parser.add_argument('--extra_name', type=str, default='Extra name to identify the model')

    # TODO: Que pasa con esto?
    parser.add_argument('--max_mean', type=int, default=10)
    parser.add_argument('--max_var', type=int, default=1)
    # parser.add_argument('--save_file', type=str, default='1-VAE-model-MNIST')

    args = parser.parse_args()

    return check_args(args)


def check_args(args):
    '''
    This method check the values provided are correct
    '''
    try:
        assert args.epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --z_dim
    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')

    try:
        assert('model' in args.model_name)
    except:
        print('bad model name')

    return args
