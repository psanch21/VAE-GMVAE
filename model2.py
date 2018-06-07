# -*- coding: utf-8 -*-

# TODO: file copied from model1.py
# CNN encoder model
# hidden_dim is the number of neurons of the first hidden layer


import tensorflow as tf
import numpy as np

import aux.utils as utils


# kernel_init = tf.truncated_normal_initializer(stddev=0.01)
kernel_init = tf.contrib.layers.variance_scaling_initializer()
bias_init = tf.constant_initializer(0.0)
act_func = tf.nn.relu

def encoder(input_, output_dim, name_scope, args, reuse, rate):

    with tf.variable_scope(name_scope + '_mean'):
        enc_mean, h_enc_mean = utils.conv_network(input_, output_dim, args['hidden_dim'], args['num_layers'], reuse, rate)

    with tf.variable_scope(name_scope + '_logvar'):
        enc_logvar, h_enc_logvar = utils.conv_network(input_, output_dim,args['hidden_dim'], args['num_layers'], reuse, rate)

    return enc_mean, enc_logvar


def decoder(input_, output_dim, name_scope, args, reuse, rate):
    hidden_dim = args['hidden_dim']

    with tf.variable_scope(name_scope):

        dec_output, _ = utils.deconv_network(input_, output_dim, args['hidden_dim'], 2, reuse, rate, out_act=tf.nn.sigmoid)

    return dec_output


def new_samples(num_imgs, z_dim, output_dim, name_scope, args, rate):
    hidden_dim = args['hidden_dim']
    # Sample from N(0,1)
    eps_sample = tf.random_normal((num_imgs, z_dim), 0, 1, dtype=tf.float32)

    with tf.variable_scope(name_scope):
        samples, _ = utils.deconv_network(eps_sample, output_dim, args['hidden_dim'], 2, True, rate, out_act=tf.nn.sigmoid)
    return samples


'''
    shape = [None, data_Dim] which means the data is flat.
    input should be between [0,1]
'''


def graph(shape, reuse, rate, args):
    print('\nDefining Computation Graph...')
    data_dim = shape[1] * shape[2] * shape[3]
    z_dim = args['z_dim']
    sigma_recons = args['sigma']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    num_imgs = args['num_imgs']

    print('[*] Input shape:', shape)
    with tf.name_scope('input'):
        data = tf.placeholder(tf.float32, shape=shape, name='x-input')

    x = data

    print('\n[*] Defining Encoder...')

    # Encoder mean log diagonal covariance
    enc_mean, enc_logvar = encoder(x, z_dim, 'encoder', args,reuse, rate)

    # Reparameterization trick. We sample from a Enc_hdim-dimensional independent Gaussian distribution
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    z = enc_mean + tf.multiply(tf.sqrt(tf.exp(enc_logvar)), eps)

    utils.variable_summary(z, 'z_latent')
    print('\n[*] Defining Decoder...')
    dec_mean = decoder(z, shape, 'decoder_mean', args, reuse, rate)

    print('\n[*] Defining Sample operation...')
    # REMEMBER WE ARE REUSING THE DECODER NETWORK
    samples = new_samples(num_imgs, z_dim, shape, 'decoder_mean', args, rate)

    print('\n[*] Defining Loss Functions and Optimizer...')

    # Reconstruction error (decoder variance is constant! given by sigma_reconstruction parameter)
    with tf.name_scope('loss_recons'):
        loss_reconstruction = -0.5 / sigma_recons * tf.reduce_sum(tf.squared_difference(dec_mean, x), [1, 2, 3])

    loss_reconstruction_m = -tf.reduce_mean(loss_reconstruction)

    # Regularization Term: KL(mu(),sigma()||N(0,1)) KL (Q(z|X)||P(z))
    with tf.name_scope('loss_kl'):
        loss_KL = 0.5 * tf.reduce_sum(tf.exp(enc_logvar) + tf.square(enc_mean) - 1 - enc_logvar, 1)

    loss_KL_m = tf.reduce_mean(loss_KL)

    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(loss_reconstruction - loss_KL)

    with tf.variable_scope('optimizer', reuse=reuse):
        # Adam offers several advantages over the simple tf.train.GradientDescentOptimizer.
        # Foremost is that it uses moving averages of the parameters (momentum);
        # This enables Adam to use a larger effective step size, and the algorithm will converge to this step size without fine tuning.
        # The main down side of the algorithm is that Adam requires more computation to be performed for each parameter
        # in each training step (to maintain the moving averages and variance, and calculate the scaled gradient);
        # and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter).
        # A simple tf.train.GradientDescentOptimizer could equally be used in your MLP, but would require more hyperparameter tuning before it would converge as quickly.
        # train_step
        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    print('[*] Defining Summary operations...')
    with tf.name_scope('loss'):
         loss_summary = tf.summary.scalar("Total", loss)
         loss_kl_summary = tf.summary.scalar("KL", loss_KL_m)
         loss_r_summary = tf.summary.scalar("Reconstruction", loss_reconstruction_m)


    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    tf_nodes = {'data': data,
                'x': x,
                'eps': eps,
                'z': z,
                'enc_mean': enc_mean,
                'enc_logvar': enc_logvar,
                'dec_mean': dec_mean,
                'loss_reconstruction_m': loss_reconstruction_m,
                'loss_KL_m': loss_KL_m,
                'loss': loss,
                'optim': optim,
                'samples': samples,
                'loss_summary': loss_summary,
                'loss_kl_summary': loss_kl_summary,
                'loss_r_summary': loss_r_summary,
                'summary_op': summary_op}

    return tf_nodes
