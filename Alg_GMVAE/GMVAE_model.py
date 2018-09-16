#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:24:14 2018

@author: psanch
"""
from base.base_model import BaseModel
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from GMVAE_graph import GMVAEGraph
from GMVAECNN_graph import GMVAECNNGraph

from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
import sys

import utils.utils as utils
import utils.constants as const

class GMVAEModel(BaseModel):
    def __init__(self,network_params,sigma=0.001, sigma_act=tf.nn.softplus,
                 transfer_fct= tf.nn.relu,learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),batch_size=32,
                 drop_rate=0., epochs=200, checkpoint_dir='', 
                 summary_dir='', result_dir='', restore=0, model_type=0):
        super().__init__(checkpoint_dir, summary_dir, result_dir)
        
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.z_file = result_dir + '/z_file'
    
        self.restore = restore
        
        
        # Creating computational graph for train and test
        self.graph = tf.Graph()
        with self.graph.as_default():
            if(model_type == const.GMVAE):
                self.model_graph = GMVAEGraph(network_params,sigma, sigma_act,
                                          transfer_fct,learning_rate, kinit,batch_size,
                                          reuse=False)
            if(model_type == const.GMVAECNN):
                self.model_graph = GMVAECNNGraph(network_params,sigma, sigma_act,
                                          transfer_fct,learning_rate, kinit,batch_size,
                                          reuse=False)          

            self.model_graph.build_graph()
            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            # model_vars = tf.trainable_variables()
            # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
            
    
    def train_epoch(self, session,logger, data_train, beta=1):
        loop = tqdm(range(data_train.num_batches(self.batch_size)))
        losses = []
        recons = []
        cond_prior = []
        KL_w = []
        y_prior = []
        L2_loss = []
        
        for _ in loop:
            batch_x = next(data_train.next_batch(self.batch_size))
            # loss_aux, recon_aux, cond_prior_aux, KL_w_aux, y_prior_aux, L2_loss_aux
            loss_list = self.model_graph.partial_fit(session, batch_x, beta, self.drop_rate)
            losses.append(loss_list[0])
            recons.append(loss_list[1])
            cond_prior.append(loss_list[2])
            KL_w.append(loss_list[3])
            y_prior.append(loss_list[4])
            L2_loss.append(loss_list[5])
        
        losses = np.mean(losses)
        recons = np.mean(recons)
        cond_prior = np.mean(cond_prior)
        KL_w = np.mean(KL_w)
        y_prior = np.mean(y_prior)
        L2_loss = np.mean(L2_loss)      

        
        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': losses,
            'recons_loss': recons,
            'CP_loss': cond_prior,
            'KL_w_loss': KL_w,
            'y_p_loss': y_prior,
            'L2_loss': L2_loss
        }
        
        logger.summarize(cur_it, summaries_dict=summaries_dict)
        
        return losses, recons, cond_prior, KL_w, y_prior, L2_loss
        
    def valid_epoch(self, session, logger, data_valid,beta=1):
        # COMPUTE VALID LOSS
        loop = tqdm(range(data_valid.num_batches(self.batch_size)))
        losses = []
        recons = []
        cond_prior = []
        KL_w = []
        y_prior = []
        L2_loss = []

        for _ in loop:
            batch_x = next(data_valid.next_batch(self.batch_size))
            loss_list = self.model_graph.evaluate(session, batch_x, beta)
            
            losses.append(loss_list[0])
            recons.append(loss_list[1])
            cond_prior.append(loss_list[2])
            KL_w.append(loss_list[3])
            y_prior.append(loss_list[4])
            L2_loss.append(loss_list[5])

        losses = np.mean(losses)
        recons = np.mean(recons)
        cond_prior = np.mean(cond_prior)
        KL_w = np.mean(KL_w)
        y_prior = np.mean(y_prior)
        L2_loss = np.mean(L2_loss)      

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': losses,
            'recons_loss': recons,
            'CP_loss': cond_prior,
            'KL_w_loss': KL_w,
            'y_p_loss': y_prior,
            'L2_loss': L2_loss
        }
        logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict)
        
        return losses, recons, cond_prior, KL_w, y_prior, L2_loss
        
    def train(self, data_train, data_valid, enable_es=1):
        
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)
            
            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            early_stopping = EarlyStopping()
            
            if(self.restore==1 and self.load(session, saver) ):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)      
            else:
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()
                
                   
            if(self.model_graph.cur_epoch_tensor.eval(session) ==  self.epochs):
                return
            
            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(session), self.epochs + 1, 1):
        
                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch
                # beta=utils.sigmoid(cur_epoch- 50)
                beta = 1.
                losses, recons, cond_prior, KL_w, y_prior, L2_loss = self.train_epoch(session, logger, data_train, beta=beta)
                train_string = 'TRAIN | Loss: ' + str(losses) + \
                            ' | Recons: ' + str(recons) + \
                            ' | CP: ' + str(cond_prior) + \
                            ' | KL_w: ' + str(KL_w) + \
                            ' | KL_y: ' + str(y_prior) + \
                            ' | L2_loss: '+  str(L2_loss)
                            
                if np.isnan(losses):
                    print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    print('Recons: ', recons)
                    print('CP: ', cond_prior)
                    print('KL_w: ', KL_w)
                    print('KL_y: ', y_prior)
                    sys.exit()
                    
                loss_val, recons, cond_prior, KL_w, y_prior, L2_loss = self.valid_epoch(session, logger, data_valid, beta=beta)
                valid_string = 'VALID | Loss: ' + str(loss_val) + \
                            ' | Recons: ' + str(recons) + \
                            ' | CP: ' + str(cond_prior) + \
                            ' | KL_w: ' + str(KL_w) + \
                            ' | KL_y: ' + str(y_prior) + \
                            ' | L2_loss: '+  str(L2_loss)
                            
                print(train_string)
                print(valid_string)
                
                if(cur_epoch>0 and cur_epoch % 10 == 0):
                    self.save(session, saver, self.model_graph.global_step_tensor.eval(session))
                    
                session.run(self.model_graph.increment_cur_epoch_tensor)
                
                #Early stopping
                if(enable_es==1 and early_stopping.stop(loss_val)):
                    print('Early Stopping!')
                    break
                    
        
            self.save(session,saver, self.model_graph.global_step_tensor.eval(session))

        return
    
    def generate_samples(self, data,num_batches=20):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            x_batch = data.random_batch(self.batch_size)
            x_samples,  z_samples,  w_samples = self.model_graph.generate_samples(session, x_batch, beta=1, num_batches=num_batches)
            
            return x_samples,  z_samples, w_samples
            
            
    def reconstruct_input(self, data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            x_batch, x_labels = data.random_batch_with_labels(self.batch_size)
            x_recons, z_recons, w_recons, y_recons = self.model_graph.reconstruct_input(session, x_batch, beta=1)
            return x_batch, x_labels, x_recons, z_recons, w_recons, y_recons 
        
    def generate_embedding(self, data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            x_batch, x_labels = data.random_batch_with_labels(self.batch_size)
            x_recons, z_recons = self.model_graph.reconstruct_input(session, x_batch, beta=1)
            return x_batch, x_labels, x_recons,  z_recons   
        
    '''  ------------------------------------------------------------------------------
                                         DISTRIBUTIONS
        ------------------------------------------------------------------------------ '''
        
    def print_parameters():
        print('')
            
    