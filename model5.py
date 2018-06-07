

# -*- coding: utf-8 -*-

'''
This file contains the GMVAE model with the following considerations:
    * Bias of last layer of Qz_x_mean are initialize using tf.truncated_normal_initializer to motivate the clustering nature
    * PDFs involving images are implented using ConvNets
'''
# n fully connected layers at both encoding and decoding
# hidden_dim is the number of neurons of the first hidden layer

import sys
import tensorflow as tf
import numpy as np

import aux.utils as utils
import aux.config as config

# args = config.parse_args()

act_func_mean = None
act_func_var = tf.tanh
max_value = 1
max_value_var = 5

std_bias=0
# Q(z|x) \sim N(\mu,\sigma^2)
def Qz_x(x, z_dim, args, reuse, rate):

    # MEAN
    with tf.variable_scope('Qz_x_mean'):

        # Notice the bias is initialize with tf.truncated_normal
        z_mean, h_z_mean = utils.conv_network(x, z_dim,args['hidden_dim'], args['num_layers'], reuse, rate, out_act= act_func_mean, std_bias=std_bias)
        z_mean = tf.scalar_mul(max_value,z_mean)
    # VARIANCE
    with tf.variable_scope('Qz_x_var'):
        z_var_aux, h_z_var = utils.conv_network(x, z_dim,args['hidden_dim'], args['num_layers'], reuse, rate, out_act= act_func_var)
        z_var_aux = tf.scalar_mul(max_value_var,z_var_aux)
        z_var = utils.sigma(z_var_aux)

    return z_mean, z_var

# Q(w|x) \sim N(0,1)
def Qw_x(x, w_dim, args, reuse, rate):

    # MEAN
    with tf.variable_scope('Qw_x_mean'):
        w_mean, h_w_mean = utils.conv_network(x, w_dim,args['hidden_dim'], args['num_layers'], reuse, rate)
    # VARIANCE
    with tf.variable_scope('Qw_x_var'):
        w_var_aux, h_w_logvar = utils.conv_network(x, w_dim, args['hidden_dim'], args['num_layers'], reuse, rate, out_act=tf.tanh)
        w_var = utils.sigma(w_var_aux)

    return w_mean, w_var



# P(z|w,y) \sim N(\mu,\sigma^2)
def Pz_wy(w, z_dim, reuse, rate,K_clusters, hidden_dim=64, num_layers=2):

    with tf.variable_scope('Pz_wy'):
        h_out, _ = utils.dense_network(w, hidden_dim, hidden_dim, num_layers-1,reuse, rate, out_act= tf.nn.relu)

        z_means = list()
        with tf.variable_scope('mean'):
            for i in range(K_clusters):
                z_mean = utils.dense_dropout(h_out, z_dim, 'dense_' + str(i), rate, act_func=act_func_mean,  bias_init=tf.truncated_normal_initializer(stddev=std_bias), reuse=reuse)
                z_mean = tf.scalar_mul(max_value,z_mean)
                z_means.append(z_mean)
            # z_means = tf.stack(z_means)

        z_vars = list()
        with tf.variable_scope('var'):
            for i in range(K_clusters):
                z_var_aux = utils.dense_dropout(h_out, z_dim, 'dense_' + str(i), rate, act_func=act_func_var, reuse=reuse)
                z_var_aux = tf.scalar_mul(max_value_var,z_var_aux)
                z_var = utils.sigma(z_var_aux)
                z_vars.append(z_var)
            # z_logvars = tf.stack(z_logvars)


    return z_means, z_vars

def Py_zw(z, w, z_dim, reuse, rate,K_clusters,hidden_dim=64, num_layers=2):
    with tf.variable_scope('Py_zw', reuse=reuse):
        zw = tf.concat([z, w],1, name='zw_concat')
        py_logit, h_py_logit = utils.dense_network(zw, K_clusters, hidden_dim, num_layers,reuse, rate)
    return py_logit # [batch_size, K]


# P(x|z)
# TODO: Deconvolutional network
def Px_z(z, data_dim, hidden_dim, reuse, rate):

    with tf.variable_scope('Px_z', reuse=reuse):
        x_mean, _ = utils.deconv_network(z, data_dim, hidden_dim, 2, reuse, rate, out_act=tf.nn.sigmoid)

    return x_mean


# TODO: Not implmenented
def new_samples(num_imgs,z_dim, w_dim, output_dim, args, rate):
    # Sample from N(0,1)
    eps_sample = tf.random_normal((num_imgs, w_dim), 0, 1, dtype=tf.float32)

    z_means, z_logvars  = Pz_wy(eps_sample, z_dim, True, rate,args['K_clusters'], hidden_dim=args['hidden_dim'], num_layers=args['num_layers'])
    samples = list()
    for K in range(args['K_clusters']):
        samples_aux = Px_z(z_means[K], output_dim, args['hidden_dim'], True, rate)
        samples.append(samples_aux)


    return samples


'''
    shape = [None, w,h,channels]
    input should be between [0,1]
'''



'''  ------------------------------------------------------------------------------
                                    GRAPH GMVAE
------------------------------------------------------------------------------   '''
def graph(shape, reuse, rate, args):

    data_dim = shape[1]*shape[2]*shape[3]
    z_dim = args['z_dim']
    w_dim = args['w_dim']
    K_clusters = args['K_clusters']
    hidden_dim = args['hidden_dim']
    num_layers = args['num_layers']
    sigma_recons = args['sigma']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    num_imgs = args['num_imgs']

    with tf.variable_scope('input', reuse=reuse):
        data = tf.placeholder(tf.float32, shape=shape, name='x-input')

    # We reshape the input to use it as input of fully connected layer
    x = data

    print('\n[*] Defining Qz_x...')
    z_mean, z_var = Qz_x(x, z_dim, args, reuse,rate)
    z_logvar = tf.log(z_var)

    print('\n[*] Defining Qw_x...')
    w_mean, w_var = Qw_x(x, w_dim, args, reuse,rate)
    w_logvar = tf.log(w_var)

    # z_mean, z_logvar, w_mean, w_logvar = Qzw_x(x, z_dim, w_dim,  args, reuse, rate)

    print('\n[*] Sampling z...')
    # Reparameterization trick. We sample from a Enc_hdim-dimensional independent Gaussian distribution
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(z_var), eps))
    # tf.summary.scalar('z', tf.reduce_mean(z))
    utils.variable_summary(z, 'z_latent')

    print('\n[*] Sampling w...')
    # Reparameterization trick. We sample from a Enc_hdim-dimensional independent Gaussian distribution
    eps = tf.random_normal((batch_size, w_dim), 0, 1, dtype=tf.float32)
    w = tf.add(w_mean, tf.multiply(tf.sqrt(w_var), eps))
    # tf.summary.scalar('z', tf.reduce_mean(z))
    utils.variable_summary(w, 'w_latent')

    print('\n[*] Defining Pz_wy...')
    z_means, z_vars = Pz_wy(w, z_dim, reuse, rate,K_clusters, hidden_dim=hidden_dim, num_layers=num_layers )

    z_vars_stack = tf.stack(z_vars)
    z_logvars_stack = tf.log(z_vars_stack) # [K, batch_size, z_dim]
    z_means_stack = tf.stack(z_means) # [K, batch_size, z_dim]

    print('\n[*] Defining Py_zw...')
    py_logit = Py_zw(z, w, z_dim, reuse, rate,K_clusters, hidden_dim=hidden_dim, num_layers=num_layers)
    py = tf.nn.softmax(py_logit)
    # Add small constant to avoid tf.log(0)
    log_py = tf.log(1e-10 + py)

    print('\n[*] Defining Px_z...')
    x_decoded_mean = Px_z(z, shape, hidden_dim, reuse, rate)


    print('\n[*] Defining Sampling...')
    # REMEMBER WE ARE REUSING THE DECODER NETWORK
    samples = new_samples(num_imgs,z_dim, w_dim, shape, args, rate)


    print('[*] Defining Loss Functions and Optimizer...')

    # Reconstruction error (decoder variance is constant! given by sigma_reconstruction parameter)
    with tf.name_scope('reconstruct'):
        reconstruction = -0.5 / sigma_recons * tf.reduce_sum(tf.square(x_decoded_mean - x),[1, 2, 3])

    loss_reconstruction_m = -tf.reduce_mean(reconstruction)

    # Regularization Term:
    with tf.name_scope('cond_prior'):
        # TODO: tf.div or tf.divide

        # Shape  tf.reduce_sum(z_logvar,1): [batch_size]
        logq = -0.5 * tf.reduce_sum(z_logvar,1) - 0.5 * tf.reduce_sum(tf.divide(tf.square(z - z_mean), z_var), 1)
        # Shape z_logvars_stack = tf.stack(z_logvars): [K, batch_size, z_dim]
        # Shape b = tf.reduce_sum(z_logvars_stack,2): [K, batch_size]
        # Shape log_det_sigma = tf.transpose(b) : [batch_size, K]
        # Shape y_logits: [batch_size, K]
        # Shape tf.reduce_sum(tf.multiply(y_logits,log_det_sigma),1) : [batch_size]

        # TODO: Que z usar?
        z_wy = tf.expand_dims(z, 2)

        z_wy = tf.tile( z_wy , [1,1, K_clusters])# [batch_size, z_dim, K]
        z_wy = tf.transpose(z_wy,perm=[2,0,1])  # [K, batch_size, z_dim]
        log_det_sigma =  tf.transpose(tf.reduce_sum(z_logvars_stack,2)) # [batch_size, K ]

        # Shape a = tf.squared_difference(z_wy, z_means_stack): [K, batch_size, z_dim]
        # Shape b = tf.divide(a, tf.exp(z_logvars_stack)): [K, batch_size, z_dim]
        # Shape tf.reduce_sum (b, [0,2])  : [batch_size]

        aux = tf.divide(tf.square(z_wy - z_means_stack), z_vars_stack) # [K, batch_size, z_dim]
        aux = tf.reduce_sum(aux, 2) # [K, batch_size]
        aux = tf.transpose(aux) # [batch_size, K]
        aux = tf.multiply(py, aux) # [batch_size, K]
        aux = tf.reduce_sum(aux,1) # [batch_size]
        logp = -0.5 * tf.reduce_sum(tf.multiply(py, log_det_sigma),1) -0.5 * aux
        cond_prior = logq - logp

    cond_prior_m = tf.reduce_mean(cond_prior)

    # Regularization Term:
    with tf.name_scope('w_prior'):
        KL_w = 0.5 * tf.reduce_sum(w_var + tf.square(w_mean) - 1  - w_logvar, 1)

    KL_w_m = tf.reduce_mean(KL_w)

    # Regularization Term:
    with tf.name_scope('y_prior'):
        # Shape logy_logits : [batch_size,10]
        y_prior  = -np.log(K_clusters, dtype='float32')  -1/K_clusters * tf.reduce_sum(log_py, axis=1)
    y_prior_m = tf.reduce_mean(y_prior)

    loss_KL_w_m = tf.reduce_mean(KL_w)
    with tf.name_scope('loss_total'):
        loss = -tf.reduce_mean(reconstruction - cond_prior - KL_w - y_prior )

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

         loss_r_summary = tf.summary.scalar("Reconstruction", loss_reconstruction_m)
         loss_cp_summary = tf.summary.scalar("Conditional_Prior", cond_prior_m)
         loss_wp_summary = tf.summary.scalar("W_prior", KL_w_m)
         loss_yp_summary = tf.summary.scalar("Y_prior", y_prior_m)
    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    tf_nodes = {'data': data,
                'x_decoded_mean': x_decoded_mean,
                'py': py,
                'z_mean': z_mean,
                'w_mean': w_mean,
                'loss_reconstruction_m': loss_reconstruction_m,
                'cond_prior_m': cond_prior_m,
                'KL_w_m': KL_w_m,
                'y_prior_m': y_prior_m,
                'loss': loss,
                'optim': optim,
                'samples': samples,
                'summary_op': summary_op}

    return tf_nodes
