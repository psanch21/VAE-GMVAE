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

from scipy.io import loadmat

'''  ------------------------------------------------------------------------------
                                    FOLDER/FILE METHODS
 ------------------------------------------------------------------------------ '''

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    return

def open_log_file(filename, args):
    '''
    Open a file and writes the first line if it does not exists
    '''
    if(os.path.isfile(filename) ):
        return

    with open(filename, 'w+') as logfile:
        my_string = ''
        for arg in args[:-1]:
            my_string+= arg +';'

        my_string+= args[-1] + '\n'
        logfile.write(my_string)
    return

def write_log_file(filename, args):
    '''
    Write a line to a file with elements separated by commas.
    '''
    if(not os.path.isfile(filename) ):
        return

    with open(filename, 'a+') as logfile:
        my_string = ''
        for arg in args[:-1]:
            my_string+= arg +';'

        my_string+= args[-1] + '\n'
        logfile.write(my_string)
    return



def save_img(fig, image_name, model_name, result_dir):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fig.savefig(os.path.join(
        result_dir, model_name + '_' + image_name + '.png'))



'''  ------------------------------------------------------------------------------
                                    PRINT METHODS
 ------------------------------------------------------------------------------ '''

def printt(string, log):
    if(log):
        print(string)

def print_loss(epoch, start_time, avg_loss, avg_loss_recons, avg_loss_KL, diff=0):
    retval = "Epoch: [%2d]  time: %4.4f, loss: %.8f, rec: %.8f, kl: %.8f, diff: %.8f" % (epoch, time.time() - start_time, avg_loss, avg_loss_recons, avg_loss_KL, diff)
    print(retval)
    return retval

def print_loss_GMVAE(epoch, start_time, avg_loss, avg_loss_recons, avg_loss_cp, avg_loss_wp, avg_loss_yp, diff=0):
    retval = "Epoch: [%2d]  time: %4.4f, loss: %.8f, rec: %.8f, cond_prior: %.8f,w_prior: %.8f,y_prior: %.8f, diff: %.8f" % (epoch, time.time() - start_time, avg_loss, avg_loss_recons, avg_loss_cp,avg_loss_wp, avg_loss_yp, diff)
    print(retval)
    return retval


def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + '\n'


def get_params(args):

    retval = ''
    for key in args:
        retval += '\t' + str(key) + ':' + str(args[key]) + '\n'
    return retval


'''  ------------------------------------------------------------------------------
                                    TENSORFLOW METHODS
 ------------------------------------------------------------------------------ '''

def get_writer(model_dir, tr_val, log_dir):
    direct = log_dir + '/' + model_dir + '/' + tr_val
    clean_folder(direct)
    # clean_folder(log_dir + '/' + model_dir + '/' + tr_val)
    writer = tf.summary.FileWriter(direct, tf.get_default_graph())
    return writer, direct


def load(sess, saver, model_dir, checkpoint_dir,logs = True):

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    printt(" \n[*] Restore model: " + model_dir, logs)
    printt(" [*] Reading checkpoints...", logs)

    check_folder(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(
            checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

        printt(" [*] Success to read {}".format(ckpt_name), logs)
        return True, counter
    else:
        printt(" [*] Failed to find a checkpoint", logs)
        return False, 0

def save(sess, saver, model_dir, checkpoint_dir, step):

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    # clean_folder(checkpoint_dir)

    saver.save(sess, os.path.join(
        checkpoint_dir, model_dir + '.model'), global_step=step)
    return


'''  ------------------------------------------------------------------------------
                                    DATA METHODS
 ------------------------------------------------------------------------------ '''

def load_data(dataset_name, model_name):
    print('\n Loading data...')
    flat = False

    if(dataset_name == 'MNIST'):
        return load_MNIST(flat)
    elif(dataset_name == 'FREY'):
        return load_FREY(flat)

    return None

def load_FREY(flat):
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/aux/'
    mat = loadmat(data_path + 'frey_rawface.mat')
    data = mat['ff']
    data = np.transpose(data) # [num_images, dimension]
    data = np.array(data, dtype=np.float32)
    for i in range(data.shape[0]):
        min_value = np.min(data[i,:])
        max_value = np.max(data[i,:])
        num = (data[i,:] - min_value)
        den = (max_value - min_value)
        data[i,:] = num/den

    data_dim = data.shape[1]
    num_images = data.shape[0]
    train_size = int(num_images*0.8)
    valid_size = int(num_images*0.1)
    test_size = num_images - train_size - valid_size

    x_train = data[:train_size]
    x_valid = data[train_size:(train_size+valid_size)]
    x_test = data[(train_size+valid_size):]

    x_train = np.reshape(x_train, [-1, 28, 20, 1])
    x_valid = np.reshape(x_valid, [-1, 28, 20, 1])
    x_test = np.reshape(x_test, [-1, 28, 20, 1])

    x_train_labels = np.zeros(x_train.shape[0])
    x_valid_labels = np.zeros(x_valid.shape[0])
    x_test_labels = np.zeros(x_test.shape[0])

    train_dataset = Dataset(x_train, x_train_labels, flat)
    valid_dataset = Dataset(x_valid, x_valid_labels, flat)
    test_dataset = Dataset(x_test, x_test_labels, flat)

    print('Train Data: ', train_dataset.data.shape)
    print('Valid Data: ', valid_dataset.data.shape)
    print('Test Data: ', test_dataset.data.shape)

    return train_dataset, valid_dataset, test_dataset, data_dim


def load_MNIST(flat):
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/aux/'
    data = input_data.read_data_sets(data_path + 'MNIST_data', one_hot=False)
    x_train_aux = data.train.images
    x_test = data.test.images
    data_dim = data.train.images.shape[1]
    n_train = data.train.images.shape[0]

    train_size = int(n_train * 0.8)
    valid_size = n_train - train_size
    x_valid, x_train = merge_datasets(x_train_aux, data_dim, train_size, valid_size)
    print('Data loaded. ', time.localtime().tm_hour,
          ':', time.localtime().tm_min, 'h')
    # logs.write('\tData loaded ' + str(time.localtime().tm_hour) +':' + str(time.localtime().tm_min) + 'h\n')

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_valid = np.reshape(x_valid, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])


    train_dataset = Dataset(x_train, data.train.labels, flat)
    valid_dataset = Dataset(x_valid, data.train.labels, flat)
    test_dataset = Dataset(x_test, data.test.labels, flat)

    print('Train Data: ', train_dataset.data.shape)
    print('Valid Data: ', valid_dataset.data.shape)
    print('Test Data: ', test_dataset.data.shape)

    return train_dataset, valid_dataset, test_dataset, data_dim



def merge_datasets(data, data_dim, train_size, valid_size=0):
    valid_dataset = np.ndarray((valid_size, data_dim), dtype=np.float32)
    train_dataset = np.ndarray((train_size, data_dim), dtype=np.float32)

    np.random.shuffle(data)

    if valid_dataset is not None:
        valid_dataset = data[:valid_size, :]

    train_dataset = data[valid_size:, :]

    return valid_dataset, train_dataset


'''  ------------------------------------------------------------------------------
                                    ACTIVATION METHODS
 ------------------------------------------------------------------------------ '''

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# We add 0.1 to avoid the variance is smaller

def sigma(tensor):
    return tf.add(tf.nn.softplus(tensor), 0.1)
'''  ------------------------------------------------------------------------------
                                    LAYERS METHODS
 ------------------------------------------------------------------------------ '''
# tf.truncated_normal_initializer(stddev=0.03)

def dense_dropout(input_, output_dim, name, rate, act_func=tf.nn.relu, kernel_init =tf.variance_scaling_initializer(), bias_init=tf.constant_initializer(0.0), reuse=None):

    h = tf.layers.dense(inputs=input_, units=output_dim, activation=act_func, kernel_initializer=kernel_init, name=name, reuse=reuse, bias_initializer=bias_init)
    out = tf.layers.dropout(h,rate=rate,name=name+'_dropout')
    print('[*] Layer (', h.name, ') output shape:', h.get_shape().as_list())

    with tf.variable_scope(name, reuse=True):
        variable_summary(tf.get_variable('kernel'), 'kernel')
        variable_summary(tf.get_variable('bias'), 'bias')
    return out

def dense(input_, output_dim, name, act_func=tf.nn.relu, kernel_init =tf.variance_scaling_initializer(), bias_init=tf.constant_initializer(0.0), reuse=None):

    h = tf.layers.dense(inputs=input_, units=output_dim, activation=act_func, kernel_initializer=kernel_init, name=name, reuse=reuse, bias_initializer=bias_init)
    print('[*] Layer (', h.name, ') output shape:', h.get_shape().as_list())
    return h


def conv(input_, filters, k_size, strides, padding, name, act_func=tf.nn.relu, kernel_init = tf.contrib.layers.xavier_initializer(), bias_init = tf.constant_initializer(0.0), reuse=None ):

    conv = tf.layers.conv2d(inputs=input_, filters=filters, kernel_size=k_size, strides=strides, padding=padding, activation=act_func, kernel_initializer=kernel_init, bias_initializer=bias_init, name=name, reuse=reuse)
    print('[*] Layer (',conv.name, ') output shape:', conv.get_shape().as_list())

    return conv

def deconv(input_, filters, k_size, strides, padding, name, act_func=tf.nn.relu, kernel_init = tf.contrib.layers.variance_scaling_initializer(), bias_init = tf.constant_initializer(0.0), reuse=None ):
    deconv = tf.layers.conv2d_transpose(input_, filters,k_size, strides=strides, padding=padding, activation=act_func, kernel_initializer=kernel_init, bias_initializer=bias_init, name=name, reuse=reuse)
    print('[*] Layer (',deconv.name, ') output shape:', deconv.get_shape().as_list())

    return deconv

def max_pool(input_, pool_size, strides, name):

    # Pooling Layer #1 output = [batch_size,14, 14,32]
    pool = tf.layers.max_pooling2d(inputs=input_, pool_size=[2, 2], strides=2, name=name )
    print('[*] Layer (',pool.name, ') output shape:', pool.get_shape().as_list())

    return pool


'''  ------------------------------------------------------------------------------
                                    NETWORKS METHODS
 ------------------------------------------------------------------------------ '''
def dense_network_biased(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None, std_bias=1):
    output = None
    h = dict()
    print("")

    # h['H1'] = tf.layers.dense(inputs=input_network, units=hidden_dim, activation=act_func, kernel_initializer=def_init, name='layer_1', reuse=reuse)
    h['H1'] = dense_dropout(input_, hidden_dim,'dense_1', rate, reuse=reuse)


    for i in range(2, num_layers + 1):
        if(i == num_layers):
            output = dense(h['H' + str(i - 1)], output_dim, 'dense_' + str(i), act_func=out_act, bias_init=tf.truncated_normal_initializer(stddev=std_bias), reuse=reuse)
            # enc_mean = densei_dropout(h['H' + str(i - 1)], output_dim, None, def_init, 'layer_' + str(i), rate, reuse=reuse)
            with tf.variable_scope('dense_' + str(i), reuse=True):
                variable_summary(tf.get_variable('kernel'), 'kernel')
                variable_summary(tf.get_variable('bias'), 'bias')
        else:
            # h['H' + str(i)] = tf.layers.dense(inputs=h['H' + str(i - 1)], units=input_dim, activation=act_func, kernel_initializer=def_init, name='layer_' + str(i), reuse=None)
            h['H' + str(i)] = dense_dropout(h['H' + str(i - 1)], hidden_dim, 'dense_' + str(i), rate, reuse=reuse)

    return output, h

def dense_network(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None):
    output = None
    h = dict()
    print("")

    # h['H1'] = tf.layers.dense(inputs=input_network, units=hidden_dim, activation=act_func, kernel_initializer=def_init, name='layer_1', reuse=reuse)
    h['H1'] = dense_dropout(input_, hidden_dim,'dense_1', rate, reuse=reuse)

    for i in range(2, num_layers + 1):
        if(i == num_layers):
            output = dense(h['H' + str(i - 1)], output_dim, 'dense_' + str(i), act_func=out_act, reuse=reuse)
            # enc_mean = densei_dropout(h['H' + str(i - 1)], output_dim, None, def_init, 'layer_' + str(i), rate, reuse=reuse)
            with tf.variable_scope('dense_' + str(i), reuse=True):
                variable_summary(tf.get_variable('kernel'), 'kernel')
                variable_summary(tf.get_variable('bias'), 'bias')
        else:
            # h['H' + str(i)] = tf.layers.dense(inputs=h['H' + str(i - 1)], units=input_dim, activation=act_func, kernel_initializer=def_init, name='layer_' + str(i), reuse=None)
            h['H' + str(i)] = dense_dropout(h['H' + str(i - 1)], hidden_dim, 'dense_' + str(i), rate, reuse=reuse)

    if(num_layers==1):
        output = h['H1']

    return output, h

def conv_network(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None, std_bias=0):
    output = None
    h = dict()

    # Convolutional Layer #1 output = [batch_size,28, 20, 16]
    filters = 16
    kernel_size=5 #[5, 5]
    strides=1
    padding='same'
    x = conv(input_, filters, kernel_size, strides, padding, 'conv1', reuse=reuse )

# Convolutional Layer #2 output = [batch_size,28, 20, 32]
    filters = 32
    kernel_size=5 #[5, 5]
    strides=1
    padding='same'
    x = conv(input_, filters, kernel_size, strides, padding, 'conv2', reuse=reuse )

    # Convolutional Layer #3 output = [batch_size,14, 10, 32]
    filters = 32
    kernel_size=3
    strides= 2
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv3', reuse=reuse )

    # Pooling Layer #1 output = [batch_size,14, 14,32]
    # pool_size = [2,2]
    # strides = [2,2]
    # x = max_pool(x, pool_size, strides, 'pool1')

    # Convolutional Layer #4 output = [batch_size,7, 5, 64]
    filters = 64
    kernel_size=3
    strides=2
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv4', reuse=reuse )

    # Convolutional Layer #5 output = [batch_size,4, 3, 64]
    filters = 64
    kernel_size=2
    strides=2
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv5', reuse=reuse )

    # Convolutional Layer #5 output = [batch_size,4, 3, 64]
    filters = 64
    kernel_size=2
    strides=1
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv6', reuse=reuse )

    # Pooling Layer #2 output = [batch_size,7, 7,64]
    # pool_size = [2,2]
    # strides = [2,2]
    # x = max_pool(x, pool_size, strides, 'pool2')

    # Dense
    x = tf.contrib.layers.flatten(x)
    x = dense_dropout(x, hidden_dim,'dense_1', rate, reuse=reuse)
    if(std_bias<=0):
        output = dense(x, output_dim, 'dense_out', act_func=out_act, bias_init=tf.constant_initializer(0.0), reuse=reuse)
    else:
        output = dense(x, output_dim, 'dense_out', act_func=out_act, bias_init=tf.truncated_normal_initializer(stddev=std_bias), reuse=reuse)





    return output, h


def deconv_network(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None):
    output = None
    h = dict()
    aux_size = output_dim[1]//2//2
    aux_size_2 = output_dim[2]//2//2
    out_dense_dim = aux_size*aux_size_2*32

    x = dense_dropout(input_, hidden_dim, 'dense1', rate, reuse=reuse)
    x = dense_dropout(x, out_dense_dim, 'dense2', rate, reuse=reuse)
    x = tf.reshape(x, [-1,aux_size,aux_size_2,32]) # [-1, aux_size, aux_size, 128]

    # Deconvolutional Layer #1 output = [batch_size,14, 14,64]
    filters = 32
    kernel_size= 2 # [aux_size +1 , aux_size +1 ]
    strides=1
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv1', reuse=reuse )

    filters = 32
    kernel_size= 2
    strides=2
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv2', reuse=reuse )


    filters = 16
    kernel_size= 3
    strides=2
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv3', reuse=reuse )

    filters = 16
    kernel_size=5
    strides=1
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv4', reuse=reuse )

    # Convolutional Layer #2 output =[batch_size,28, 28,channel_num]
    filters = output_dim[3]
    kernel_size=5
    strides=1
    padding='same'
    output = deconv(x, filters, kernel_size, strides, padding, 'deconv5', reuse=reuse, act_func=out_act )

    return output, h



def conv_network2(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None, std_bias=0):
    output = None
    h = dict()

    # Convolutional Layer #1 output = [batch_size,28, 28, 16]
    filters = 16
    kernel_size=5 #[5, 5]
    strides=1
    padding='same'
    x = conv(input_, filters, kernel_size, strides, padding, 'conv1', reuse=reuse )

    filters = 16
    kernel_size=5 #[5, 5]
    strides=1
    padding='same'
    x = conv(input_, filters, kernel_size, strides, padding, 'conv2', reuse=reuse )

    # Convolutional Layer #2 output = [batch_size,14, 14,32]
    filters = 32
    kernel_size=3
    strides= 2
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv3', reuse=reuse )

    # Pooling Layer #1 output = [batch_size,14, 14,32]
    # pool_size = [2,2]
    # strides = [2,2]
    # x = max_pool(x, pool_size, strides, 'pool1')

    filters = 64
    kernel_size=3
    strides=2
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv4', reuse=reuse )


    filters = 64
    kernel_size=2
    strides=2
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv5', reuse=reuse )

    filters = 64
    kernel_size=2
    strides=1
    padding='same'
    x = conv(x, filters, kernel_size, strides, padding, 'conv6', reuse=reuse )

    # Pooling Layer #2 output = [batch_size,7, 7,64]
    # pool_size = [2,2]
    # strides = [2,2]
    # x = max_pool(x, pool_size, strides, 'pool2')

    # Dense

    # prev_dim = int(np.prod(x.get_shape()[1:]))
    # x = tf.reshape(x, [-1, prev_dim])
    x = tf.contrib.layers.flatten(x)
    x = dense_dropout(x, hidden_dim,'dense_1', rate, reuse=reuse)
    if(std_bias<=0):
        output = dense(x, output_dim, 'dense_out', act_func=out_act, bias_init=tf.constant_initializer(0.0), reuse=reuse)
    else:
        output = dense(x, output_dim, 'dense_out', act_func=out_act, bias_init=tf.truncated_normal_initializer(stddev=std_bias), reuse=reuse)

    return output, h


def deconv_network2(input_, output_dim,hidden_dim, num_layers, reuse, rate, out_act= None):
    output = None
    h = dict()
    aux_size = output_dim[1]//2//2
    out_dense_dim = aux_size*aux_size*32

    x = dense_dropout(input_, hidden_dim, 'dense1', rate, reuse=reuse)
    x = dense_dropout(x, out_dense_dim, 'dense2', rate, reuse=reuse)
    x = tf.reshape(x, [-1,aux_size,aux_size,32]) # [-1, aux_size, aux_size, 128]

    # Deconvolutional Layer #1 output = [batch_size,14, 14,64]
    filters = 32
    kernel_size= 2 # [aux_size +1 , aux_size +1 ]
    strides=1
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv1', reuse=reuse )

    filters = 32
    kernel_size= 2
    strides=2
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv2', reuse=reuse )


    filters = 16
    kernel_size= 3
    strides=2
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv3', reuse=reuse )

    filters = 16
    kernel_size=5
    strides=1
    padding='same'
    x= deconv(x, filters, kernel_size, strides, padding, 'deconv4', reuse=reuse )

    # Convolutional Layer #2 output =[batch_size,28, 28,channel_num]
    filters = output_dim[3]
    kernel_size=5
    strides=1
    padding='same'
    output = deconv(x, filters, kernel_size, strides, padding, 'deconv5', reuse=reuse, act_func=out_act )

    return output, h

'''  ------------------------------------------------------------------------------
                                    TENSORBOARD METHODS
 ------------------------------------------------------------------------------ '''
# Attach a lot of summaries to a Tensor


def variable_summary(var, name='summaries'):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))

        tf.summary.histogram('histogram', var)
    return
