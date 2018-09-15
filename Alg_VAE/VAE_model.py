#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:24:14 2018

@author: psanch
"""
from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
from VAE_graph import VAEGraph
from VAECNN_graph import VAECNNGraph

from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
import sys

import utils.utils as utils
import utils.constants as const

class VAEModel(BaseModel):
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
            if(model_type == const.VAE):
                self.vae_graph = VAEGraph(network_params,sigma, sigma_act,
                                          transfer_fct,learning_rate, kinit,batch_size,
                                          reuse=False)
            if(model_type == const.VAECNN):
                self.vae_graph = VAECNNGraph(network_params,sigma, sigma_act,
                                          transfer_fct,learning_rate, kinit,batch_size,
                                          reuse=False)          

            self.vae_graph.build_graph()
            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            
    
    def train_epoch(self, session,logger, data_train, beta=1):
        loop = tqdm(range(data_train.num_batches(self.batch_size)))
        losses = []
        recons = []
        cond_prior = []
        L2_loss = []
        
        for _ in loop:
            batch_x = next(data_train.next_batch(self.batch_size))
            loss, recon, cond, L2_loss_curr = self.vae_graph.partial_fit(session, batch_x, beta, self.drop_rate)
            losses.append(loss)
            recons.append(recon)
            cond_prior.append(cond)
            L2_loss.append(L2_loss_curr)
        loss_tr = np.mean(losses)
        recons_tr = np.mean(recons)
        cond_prior_tr = np.mean(cond_prior)
        L2_loss = np.mean(L2_loss)
        
        cur_it = self.vae_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': loss_tr,
            'recons_loss': recons_tr,
            'KL_loss': cond_prior_tr,
            'L2_loss': L2_loss
        }
        
        logger.summarize(cur_it, summaries_dict=summaries_dict)
        
        return loss_tr, recons_tr, cond_prior_tr, L2_loss
        
    def valid_epoch(self, session, logger, data_valid,beta=1):
        # COMPUTE VALID LOSS
        loop = tqdm(range(data_valid.num_batches(self.batch_size)))
        losses_val = []
        recons_val = []
        cond_prior_val = []
        for _ in loop:
            batch_x = next(data_valid.next_batch(self.batch_size))
            loss, recon, cond, _ = self.vae_graph.evaluate(session, batch_x, beta)
            
            losses_val.append(loss)
            recons_val.append(recon)
            cond_prior_val.append(cond)
        loss_val = np.mean(losses_val)
        recons_val = np.mean(recons_val)
        cond_prior_val = np.mean(cond_prior_val)

        cur_it = self.vae_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': loss_val,
            'recons_loss': recons_val,
            'KL_loss': cond_prior_val
        }
        logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict)
        
        return loss_val, recons_val, cond_prior_val
        
    def train(self, data_train, data_valid, enable_es=1):
        
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)
            
            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            early_stopping = EarlyStopping()
            
            if(self.restore==1 and self.load(session, saver) ):
                num_epochs_trained = self.vae_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)      
            else:
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()
                
                   
            if(self.vae_graph.cur_epoch_tensor.eval(session) ==  self.epochs):
                return
            
            for cur_epoch in range(self.vae_graph.cur_epoch_tensor.eval(session), self.epochs + 1, 1):
        
                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch
                # beta=utils.sigmoid(cur_epoch- 50)
                beta = 1.
                loss_tr, recons_tr, cond_prior_tr, L2_loss = self.train_epoch(session, logger, data_train, beta=beta)
                if np.isnan(loss_tr):
                    print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    print('Recons: ', recons_tr)
                    print('KL: ', cond_prior_tr)
                    sys.exit()
                    
                loss_val, recons_val, cond_prior_val = self.valid_epoch(session, logger, data_valid, beta=beta)
                
                print('TRAIN | Loss: ', loss_tr, ' | Recons: ', recons_tr, ' | KL: ', cond_prior_tr, ' | L2_loss: ', L2_loss)
                print('VALID | Loss: ', loss_val, ' | Recons: ', recons_val, ' | KL: ', cond_prior_val)
                
                if(cur_epoch>0 and cur_epoch % 10 == 0):
                    self.save(session, saver, self.vae_graph.global_step_tensor.eval(session))
                    z_matrix = self.vae_graph.get_z_matrix(session, data_valid.random_batch(self.batch_size))
                    np.savez(self.z_file, z_matrix)
                    
                session.run(self.vae_graph.increment_cur_epoch_tensor)
                
                #Early stopping
                if(enable_es==1 and early_stopping.stop(loss_val)):
                    print('Early Stopping!')
                    break
                    
        
            self.save(session,saver, self.vae_graph.global_step_tensor.eval(session))
            z_matrix = self.vae_graph.get_z_matrix(session, data_valid.random_batch(self.batch_size))
            np.savez(self.z_file, z_matrix)
        return
    
    def generate_samples(self, data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.vae_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            x_batch = data.random_batch(self.batch_size)
            x_samples,  z_samples = self.vae_graph.generate_samples(session, x_batch, beta=1)
            
            return x_samples,  z_samples
            
            
    def reconstruct_input(self, data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.vae_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            x_batch, x_labels = data.random_batch_with_labels(self.batch_size)
            x_recons, z_recons = self.vae_graph.reconstruct_input(session, x_batch, beta=1)
            return x_batch, x_labels, x_recons,  z_recons   
    '''  ------------------------------------------------------------------------------
                                         DISTRIBUTIONS
        ------------------------------------------------------------------------------ '''
        
    def print_parameters():
        print('')
            
    