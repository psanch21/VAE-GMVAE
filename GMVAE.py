# -*- coding: utf-8 -*-

# Variational AutoEncoder over the MNIST database using TensorFlow
import os
import sys

import tensorflow as tf
import time
from matplotlib import pyplot as plt
import numpy as np

import graph


import aux.config as config
import aux.utils as utils
import aux.visualization as vis
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.close("all")


'''  ------------------------------------------------------------------------------
                                    PARSING AND CONFIGURATION
------------------------------------------------------------------------------   '''

args = config.parse_args()
if args is None:
    exit()
print(args, '\n')

args_dict = vars(args)

class Config():
    dataset_name = args.dataset_name
    model_type = args.model_name # model4
    epochs = args.epochs
    batch_size = args.batch_size
    sigma_recons = args.sigma
    learning_rate = args.learning_rate

    z_dim = args.z_dim
    w_dim = args.w_dim
    K_clusters = args.K_clusters
    hidden_dim = args.hidden_dim

    num_layers = args.num_layers

    checkpoint_dir = 'tensorflow/' +args.checkpoint_dir
    result_dir = 'tensorflow/' +args.result_dir
    log_dir = 'tensorflow/' +args.log_dir
    data_dir = args.data_dir

    flag_train = args.train
    flag_summary = args.summary
    flag_restore = args.restore
    flag_plot = args.plot
    flag_generate = args.generate

    save_step = args.save_step

    num_imgs = args.num_imgs
    extra_name = args.extra_name
    if(extra_name == ''):
        model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(model_type, dataset_name, epochs, str(sigma_recons).replace('.', ''), z_dim, w_dim, hidden_dim, num_layers, K_clusters)
    else:
        model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(model_type, dataset_name, epochs, str(sigma_recons).replace('.', ''), z_dim, w_dim, hidden_dim, num_layers, K_clusters, extra_name)



    log_file_name =  log_dir + '/GMVAE_test.txt'
    log_summary_file_name =  log_dir + '/GMVAE_summary.txt'


config = Config()

utils.check_folder(config.checkpoint_dir)
utils.check_folder(config.result_dir)
utils.check_folder(config.log_dir)


logs = open(config.log_file_name, 'a')


cols = ['ModelName', 'ValidationLoss', 'TrainingLoss','ReconstructionLoss', 'ConditionalPrior', 'WPrior', 'YPrior' ]
utils.open_log_file(config.log_summary_file_name,cols)

logs.write('MODEL FOLDER: ' + config.model_name + '\n')
print('[*] MODEL FOLDER: ', config.model_name)
logs.write('Time(GMT): ' + utils.get_time())
logs.write('Params:\n' + utils.get_params(args_dict))

'''  ------------------------------------------------------------------------------
                                    FLAGS
------------------------------------------------------------------------------   '''

print('[*] Train Active: ', config.flag_train == 1)
print('[*] Summary Active: ', config.flag_summary == 1)
print('[*] Restore Active: ', config.flag_restore == 1)
print('[*] Plot Active: ', config.flag_plot == 1)

'''  ------------------------------------------------------------------------------
                                    LOAD DATA
------------------------------------------------------------------------------   '''

# Normalize data [0,1]: sigmoid activation
print('LOADING DATA', config.dataset_name)
x_train, x_valid, x_test, data_dim = utils.load_data(config.dataset_name, config.model_name)
n_samples = x_train._num_examples
num_batches = int(n_samples / config.batch_size)
num_batches_val = int(x_valid._num_examples / config.batch_size) # x_valid._num_examples // config.batch_size

'''  ------------------------------------------------------------------------------
                                    COMPUTATION GRAPH (Build the model)
------------------------------------------------------------------------------   '''

shape = [None]
shape.extend(x_train.data.shape[1:]) # (None,...)
sess_VAE = tf.Graph()
with sess_VAE.as_default():
    tf.set_random_seed(1234)
    # TODO: Change args_dict with config
    print('\nDefining Computation Graph...')
    tf_nodes = graph.VAE_graph(config.model_type, shape,  False, 0.4, args_dict)
    trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    print('\nDefining Computation Graph TEST...')
    tf_nodes_test = graph.VAE_graph(config.model_type, shape, True, 0.0, args_dict)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()



'''  ------------------------------------------------------------------------------
                                    TRAINING
------------------------------------------------------------------------------   '''
print('\nNumber of trainable paramters', trainable_count)
train_list = []

recons_list = []
KL_cp_list = []
KL_wp_list = []
KL_yp_list = []


valid_list = []

avg_loss_valid = 0

# with tf.Session(graph=sess_VAE) as sess:
with tf.Session(graph=sess_VAE, config=tf.ConfigProto(log_device_placement=False)) as sess:

    print('[*] Initializing Variables...')
    tf.global_variables_initializer().run()


    start_epoch = 0
    start_batch_id = 0
    counter = 1

    if(config.flag_restore == 1):
        ok_load, checkpoint_counter = utils.load(sess, saver, config.model_name, config.checkpoint_dir)
        print('check counter: ', checkpoint_counter)
        if(ok_load):
            start_epoch = checkpoint_counter
            x_train.set_epochs_completed(start_epoch)
            x_valid.set_epochs_completed(start_epoch)


            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")





    if(config.flag_train == 1):
        train_writer, _ = utils.get_writer(config.model_name, 'train', config.log_dir)
        valid_writer, _ = utils.get_writer(config.model_name, 'valid',config.log_dir)

        print('\nStarting VAE Training...')
        logs.write('Starting Training:\n')
        print('[*] Number of Epochs: ', config.epochs, '\n')

        start_time = time.time()

        diff_loss = 1000
        min_avg_loss_valid = 100000
        patience=1
        min_delta = 0.01
        epoch = x_train.get_epochs_completed()
        config.epochs= config.epochs + epoch
        while(epoch<=config.epochs and patience<=10):

            avg_loss = 0.

            avg_loss_recons = 0.
            avg_loss_cp = 0.
            avg_loss_wp = 0.
            avg_loss_yp = 0.

            # Loop over all batches
            while(x_train.get_epochs_completed()==epoch):
                # Fit training using batch data: Update Autoencoder
                batch_xs, _ = x_train.next_batch(config.batch_size, shuffle=True)

                feed = dict()
                feed[tf_nodes['data']] = batch_xs

                nodes= [tf_nodes['optim'], tf_nodes['loss'], tf_nodes['loss_reconstruction_m'], tf_nodes['cond_prior_m'], tf_nodes['KL_w_m'], tf_nodes['y_prior_m']]
                _, cost, cost_recons, cost_cp, cost_wp, cost_yp = sess.run(nodes, feed_dict=feed)
                # Compute average loss
                avg_loss += cost / n_samples * config.batch_size

                avg_loss_recons += cost_recons / n_samples * config.batch_size
                avg_loss_cp += cost_cp / n_samples * config.batch_size
                avg_loss_wp += cost_wp / n_samples * config.batch_size
                avg_loss_yp += cost_yp / n_samples * config.batch_size



            if(config.flag_summary):
                batch_xs, _ = x_train.next_batch(config.batch_size, shuffle=True)
                summary = sess.run(tf_nodes['summary_op'], feed_dict={tf_nodes['data']: batch_xs})
                train_writer.add_summary(summary, counter)

                batch_xs, _ = x_valid.next_batch(config.batch_size, shuffle=True)
                summary = sess.run(tf_nodes['summary_op'], feed_dict={tf_nodes['data']: batch_xs})
                valid_writer.add_summary(summary, counter)

            avg_loss_valid = 0.
            while(x_valid.get_epochs_completed()==epoch):
                batch_xs, _ = x_valid.next_batch(config.batch_size, shuffle=True)
                loss_valid = sess.run(tf_nodes_test['loss'], feed_dict={tf_nodes_test['data']: batch_xs})
                avg_loss_valid += loss_valid / x_valid._num_examples * config.batch_size

            if(epoch >10 and valid_list[-1] - avg_loss_valid > min_delta):
                patience=1
            elif():
                patience+=1
                print('\nPatience Level: ', patience)


            train_list.append(cost)

            recons_list.append(cost_recons)
            KL_cp_list.append(cost_cp)
            KL_wp_list.append(cost_wp)
            KL_yp_list.append(cost_yp)

            valid_list.append(avg_loss_valid)


            str_epoch = utils.print_loss_GMVAE(epoch, start_time, avg_loss, avg_loss_recons, avg_loss_cp, avg_loss_wp, avg_loss_yp, diff=avg_loss_valid - avg_loss)
            print("")
            logs.write('\t' + str_epoch + '\n')
            # Save training results
            if epoch % config.save_step == 0:
                print('[*] Saving Variables ...')
                # Save the variables to disk.
                utils.save(sess, saver, config.model_name, config.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            counter += 1
            epoch+=1

        # save model for final step
        print('[*] End Training ...')
        print('[*] Saving Variables ...\n')
        utils.save(sess, saver, config.model_name, config.checkpoint_dir, counter)

        cols = [config.model_name, str(valid_list[-1]), str(train_list[-1]), str(recons_list[-1]), str(KL_cp_list[-1]),  str(KL_wp_list[-1]),  str(KL_yp_list[-1])]
        utils.write_log_file(config.log_summary_file_name,cols)

        f_train, ax = plt.subplots()
        ax.plot(train_list, label="train")

        ax.plot(recons_list, label="recons")
        ax.plot(KL_cp_list, label="Cond Prior")
        ax.plot(KL_wp_list, label="W prior")
        ax.plot(KL_yp_list, label="Y prior")

        ax.plot(valid_list, label="valid")
        ax.legend()
        utils.save_img(f_train, 'training_summ', config.model_name, config.result_dir)



    '''  ------------------------------------------------------------------------------
                                     RESULTS
    ------------------------------------------------------------------------------   '''

    if(config.flag_generate == 1):
        print(" [*] Testing GMVAE...")
        ok_load, checkpoint_counter = utils.load(sess, saver, config.model_name, config.checkpoint_dir, False)

        if(ok_load):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        batch_xs, label_train = x_train.next_batch(config.batch_size)  # batch_xs.shape = (batch_size, w, h, channels)

        '''SAMPLING'''
        feed = dict()
        feed[tf_nodes_test['data']] = batch_xs

        nodes = list()
        nodes.append(tf_nodes_test['samples'])

        images_list = sess.run(nodes, feed_dict=feed)
        images_list = images_list[0]

        '''RECONSTRUCTION AND LATENT TRAIN'''
        nodes = list()
        nodes.append(tf_nodes_test['x_decoded_mean'])
        nodes.append(tf_nodes_test['py'])
        nodes.append(tf_nodes_test['z_mean'])
        nodes.append(tf_nodes_test['w_mean'])


        reconstructions_train, py_train, z_train, w_train = sess.run(nodes, feed_dict=feed)

        '''RECONSTRUCTION TEST'''
        batch_test, label_test = x_test.next_batch(config.batch_size)

        feed = dict()
        feed[tf_nodes_test['data']] = batch_test

        nodes = list()
        nodes.append(tf_nodes_test['x_decoded_mean'])

        reconstructions_test = sess.run(nodes, feed_dict=feed)
        reconstructions_test = reconstructions_test[0]
        #  We plot three figures:
        #   1) Samples obtained by the generative model (we first sample from p(z) and then we reconstruct)
        #   2) Reconstruction for some training images (first compressed through the encoder, then reconstructed)
        #   3) Reconstruction for some test images (not used for training)

        for idx, images_gen in enumerate(images_list):
            f = vis.plot_images_gen(x_train.get_shape(flat=False),images_gen, n_plots_axis=5, K=idx,  model_name='GMVAE')
            utils.save_img(f, 'generated_' + str(idx), config.model_name, config.result_dir)


        f2, f3 = vis.plot_summary(x_train.get_shape(flat=False), batch_xs, batch_test, reconstructions_train, reconstructions_test)

        utils.save_img(f2, 'training', config.model_name, config.result_dir)
        utils.save_img(f3, 'test', config.model_name, config.result_dir)

        '''LATENT TEST'''
        feed = dict()

        nodes = list()
        nodes.append(tf_nodes_test['py'])
        nodes.append(tf_nodes_test['z_mean'])
        nodes.append(tf_nodes_test['w_mean'])

        py_test_list = list()
        z_test_list = list()
        w_test_list = list()
        label_test_list = list()
        batch_test_list = list()
        x_test.reset()

        while(x_test.get_epochs_completed()==0):
            batch_test, label_test = x_test.next_batch(config.batch_size)
            feed[tf_nodes_test['data']] = batch_test

            py_test, z_test, w_test = sess.run(nodes, feed_dict=feed)

            py_test_list.extend(py_test)
            z_test_list.extend(z_test)
            w_test_list.extend(w_test)
            label_test_list.extend(label_test)
            batch_test_list.extend(batch_test)


        py_test = np.array(py_test_list)
        z_test = np.array(z_test_list)
        w_test = np.array(w_test_list)
        label_test = np.array(label_test_list)
        batch_test = np.array(batch_test_list)

        n_plots_axis = 5

        idx_tmp = np.arange(0, np.shape(z_test)[0])  # get all possible indexes
        np.random.shuffle(idx_tmp)  # shuffle indexe
        idx_tmp = idx_tmp[:600]

        z_test = z_test[idx_tmp]
        w_test = w_test[idx_tmp]
        label_test = label_test[idx_tmp]
        batch_test = batch_test[idx_tmp]

        print(" [*] Generating Z scatter plot...")
        f5 = vis.scatter_z_dim( z_test, label_test, perplexity=10, title="Z Scatter Test")
        utils.save_img(f5, 'z_scatter_test', config.model_name, config.result_dir)

        print(" [*] Generating W scatter plot...")
        f5 = vis.scatter_z_dim( w_test, label_test, perplexity=10, title="W Scatter Test")
        utils.save_img(f5, 'w_scatter_test', config.model_name, config.result_dir)

if(config.flag_generate == 1):
    print(" [*] Embeddings TensorBoard...")
    embed_writer, embed_logdir = utils.get_writer(config.model_name, 'embed', config.log_dir)
    vis.plot_embeddings(batch_test, label_test, z_test, embed_logdir,  x_train.get_shape(False)[1:])


'''  ------------------------------------------------------------------------------
                                    PLOT
------------------------------------------------------------------------------   '''
if(config.flag_plot == 1):
    print(' [*] Plotting results ...')
    plt.show()
else:
    plt.close("all")


logs.write('\n\t-----------------------------------------------\n')
logs.close()
