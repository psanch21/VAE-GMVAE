# -*- coding: utf-8 -*-

# n fully connected layers at both encoding and decoding
# hidden_dim is the number of neurons of the first hidden layer


import tensorflow as tf
import numpy as np

import aux.utils as utils


# def_init = tf.truncated_normal_initializer(stddev=0.01)
def_init = tf.contrib.layers.variance_scaling_initializer()
bias_init = tf.constant_initializer(0.)

n_cluster = 10
# act_func = tf.nn.relu
act_func = tf.nn.tanh

def normal(z, mean, var, z_dim):
    return 1/(tf.pow(2 * np.pi, z_dim / 2) * tf.sqrt(tf.reduce_prod(var, 1))) * tf.exp(- 1 / 2 * tf.reduce_sum(tf.square(z - mean) / var, 1))


def lognormal(z, mean, var, z_dim):
    log_var = tf.log(var)
    return -z_dim / 2 * tf.log(2 * np.pi) - 1 / 2 * tf.reduce_sum(log_var, 1) - 1 / 2 * tf.reduce_sum(tf.square(z - mean) / var, 1)

'''
def normal(z, mean, var, z_dim):
    """ Returns the multivariate normal evaluated at points z.

    Args:
        z: Tensor (batch_size, z_dim), points to be evaluated
        mean: array[batch_size, z_dim]
        var: array[batch_size, z_dim]
        z_dim: integer, dimension of the normal
    Returns:
        the normal(z)
    """
    return 1 / (tf.pow(2 * np.pi, z_dim / 2) * tf.sqrt(tf.reduce_prod(var, 1))) * tf.exp(- 1 / 2 * tf.reduce_sum(tf.square(z - mean) / var, 1))

'''

def z_dist(args):
    z_dim = args['z_dim']
    batch_size = args['batch_size']
    n_means = args['K_clusters']
    mu = np.array([0, 1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10])

    scale = tf.to_float(tf.fill([batch_size, z_dim], tf.sqrt(0.5)))
    dists = list()
    for i in range(0, n_means):
        mui = np.full([batch_size, z_dim], mu[i]).astype('f')
        #mui[:, 0] = 0.
        dists.append(tf.contrib.distributions.MultivariateNormalDiag(loc=mui, scale_diag=scale))

    return dists

def encoder(input_, output_dim, name_scope, args, reuse, rate):

    # MEAN
    with tf.variable_scope(name_scope + '_mean'):
        enc_mean, h_enc_mean = utils.dense_network(input_, output_dim, args['hidden_dim'], args['num_layers'],reuse, rate)


    # VARIANCE
    with tf.variable_scope(name_scope + '_logvar'):
          enc_logvar, h_enc_logvar = utils.dense_network(input_, output_dim, args['hidden_dim'], args['num_layers'], reuse, rate)


    return enc_mean, enc_logvar


def decoder(input_, output_dim, name_scope, args, reuse, rate):
    with tf.variable_scope(name_scope):

        dec_output, _ = utils.dense_network(input_, output_dim, args['hidden_dim'], 2, reuse, rate, out_act=tf.nn.sigmoid)

    return dec_output


def new_samples(num_imgs, output_dim, name_scope, args, rate):
    """ Generates new samples from the latent space.

    Args:
        num_imgs: integer, number of images to generate
        output_dim: dimension of a single flat image
        name_scope: context manager
        args: dictionary, contains all the arguments of the model.
    Returns:
        samples: output of decoder, num_imgs images.
    """

    hidden_dim = args['hidden_dim']
    z_dim = args['z_dim']
    # Sample from N(0,1)


    p = tf.to_float(tf.fill([n_cluster], 1 / n_cluster))
    cat = tf.distributions.Categorical(probs=p)
    aux_samples = cat.sample(num_imgs)
    # cat_samples = tf.to_float(tf.transpose(tf.stack([aux_samples] * z_dim)))
    print()
    dists = z_dist(args)
    eps_sample = dists[5].sample()

    # eps_sample = tf.random_normal((num_imgs, z_dim), 0, tf.sqrt(0.5), dtype=tf.float32)
    with tf.variable_scope(name_scope):
        samples, _ = utils.dense_network(eps_sample, output_dim, args['hidden_dim'], 2, True, rate, out_act=tf.nn.sigmoid)

    return samples


'''
    shape = [None, data_Dim] which means the data is flat.
    input should be between [0,1]
'''


def graph(shape, reuse, rate, args):
    print('\nDefining Computation Graph...')
    data_dim = shape[1]*shape[2]*shape[3]
    z_dim = args['z_dim']
    sigma_recons = args['sigma']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    num_imgs = args['num_imgs']

    with tf.name_scope('input'):
        data = tf.placeholder(tf.float32, shape=shape, name='x-input')

    x = tf.reshape(data, [-1,data_dim])
    print('[*] Defining Encoder...')

    # Encoder mean log diagonal covariance
    enc_mean, enc_logvar = encoder(x,z_dim, 'encoder', args, reuse, rate)

    # Reparameterization trick. We sample from a Enc_hdim-dimensional independent Gaussian distribution
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    z = enc_mean + tf.multiply(tf.sqrt(tf.exp(enc_logvar)), eps)


    print('[*] Defining Decoder...')
    dec_mean = decoder(z, data_dim, 'dec_mean', args, reuse, rate)

    print('[*] Defining Loss Functions and Optimizer...')

    # Reconstruction error (decoder variance is constant! given by sigma_reconstruction parameter)
    with tf.name_scope('loss_recons'):
        loss_reconstruction = -0.5 / sigma_recons * \
            tf.reduce_sum(tf.squared_difference(dec_mean, x), 1)

    loss_reconstruction_m = -tf.reduce_mean(loss_reconstruction)

    # Regularization Term: KL[N(mu(),sigma())||SUM(N(i,1))] KL

    c = tf.constant(1 / n_cluster)

    dists = z_dist(args)
    p_gmm = 0

    for i, dist in enumerate(dists):
        p_gmm += dist.prob(z)
    pz_gmm = tf.log(c) + tf.log(p_gmm)

    stddev_q = tf.sqrt(tf.exp(enc_logvar))
    dist_q = tf.contrib.distributions.MultivariateNormalDiag(loc=enc_mean, scale_diag=stddev_q)
    logq_zx = dist_q.log_prob(z)

    # tf.summary.histogram('pz_gmm', pz_gmm)
    # tf.summary.histogram('logq_zx', logq_zx)
    tf.summary.histogram('z', z)
    # Regularization Term: KL(mu(),sigma()||N(0,1)) KL (Q(z|X)||P(z))
    with tf.name_scope('loss_kl'):
        loss_KL = (logq_zx - pz_gmm)

    loss_KL_m = tf.reduce_mean(loss_KL)

    ELBO = tf.reduce_mean(loss_reconstruction - loss_KL)
    with tf.name_scope('loss'):
        loss = -ELBO

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

    print('[*] Defining Sample operation...')
    samples = new_samples(num_imgs, data_dim, 'dec_mean', args, rate)
    print('[*] Defining Summary operations...')
    loss_summary = tf.summary.scalar("loss", loss)
    loss_kl_summary = tf.summary.scalar("kl_loss", loss_KL_m)
    loss_r_summary = tf.summary.scalar("rloss", loss_reconstruction_m)

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
