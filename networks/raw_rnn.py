#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:17:53 2018

@author: psanch
"""

import tensorflow as tf
import utils.constants as const
from utils.utils import get1toT
from networks.base_raw_rnn import BaseRawRNN
from networks.dense_net import DenseNet

import utils.utils as utils

class RawRNNConcat(BaseRawRNN):
    def __init__(self, cell_type, state_dim, input_, max_time, output_dim, reuse, drop_rate_x=0.,
                 kinit=tf.contrib.layers.xavier_initializer(), 
                 bias_init=tf.constant_initializer(0.01), var_shared=False):
        super().__init__(input_, max_time, output_dim, cell_type, state_dim, reuse, kinit, bias_init)
        
    
        self.rnn_input_dim = self.input_dim + self.output_dim
        
        self.drop_rate_x = drop_rate_x
        
        self.act_out_mean = None
        self.act_out_var = tf.nn.softplus
        self.var_shared = var_shared
        
        self.output_mean, self.output_var, self.output_z = self.my_build()
    
    def my_build(self):
        output_list, state_list = self.build(self.get_loop_fn())
        outputs_mean = output_list[0]
        outputs_var = output_list[1]
        outputs_z = output_list[2]
        
        states_all_c = state_list[0]
        states_all_h = state_list[1]
        
        print('Means: ', outputs_mean.get_shape().as_list())
        print('Vars: ', outputs_var.get_shape().as_list())
        print('Sampled z: ', outputs_z.get_shape().as_list())
        print('States c: ', states_all_c.get_shape().as_list())
        print('States h: ', states_all_h.get_shape().as_list())
        return outputs_mean, outputs_var, outputs_z
        
    
    def get_output_step(self, cell_output):
        with tf.variable_scope('mean', reuse=tf.AUTO_REUSE):
            mean_net = DenseNet(input_=cell_output,
                                hidden_dim=-1, 
                                output_dim=self.output_dim, 
                                num_layers=1, 
                                transfer_fct=None,
                                act_out=self.act_out_mean, 
                                reuse=tf.AUTO_REUSE, 
                                kinit=self.kinit,
                                bias_init=self.bias_init)
            
            mean = mean_net.output
            
        with tf.variable_scope('var', reuse=tf.AUTO_REUSE):
            if(self.var_shared):
                var = utils.get_variable(self.output_dim, 'var')
                var = tf.tile(var, [self.batch_size, 1])# [batch_size, var.dim]
            else:
                var_net = DenseNet(input_=cell_output,
                                    hidden_dim=-1, 
                                    output_dim=self.output_dim, 
                                    num_layers=1, 
                                    transfer_fct=None,
                                    act_out=self.act_out_var, 
                                    reuse=tf.AUTO_REUSE, 
                                    kinit=self.kinit,
                                    bias_init=self.bias_init)
                
                var = var_net.output
        
        eps = tf.random_normal((self.batch_size, self.output_dim), 0, 1, dtype=tf.float32)
        current_z = tf.add(mean, tf.multiply(tf.sqrt(var), eps))
        return mean, var, current_z
    
    def get_next_input(self, x_time, current_z):
        with tf.variable_scope('aux', reuse=tf.AUTO_REUSE):
            aux_net = DenseNet(input_=current_z,
                                hidden_dim=-1, 
                                output_dim=self.output_dim, 
                                num_layers=1, 
                                transfer_fct=None,
                                act_out=tf.nn.sigmoid, 
                                reuse=tf.AUTO_REUSE)
            current_z = aux_net.output
        return tf.concat([tf.layers.dropout(x_time, rate=self.drop_rate_x), current_z],1)
            

    
    def get_loop_fn(self):

        inputs_ta, output_ta = self.get_tensor_arrays(self.input_)


        def loop_fn(time, cell_output, cell_state, loop_state):

            elements_finished = (time >= self.max_time)
            finished = tf.reduce_all(elements_finished)
            
            if cell_output is None:
                '''
                time == 0, used for initialization before first call to cell
                This is just to defined the desired shape of the tensors
                '''
                next_cell_state = self.cell.zero_state(self.batch_size, tf.float32)
                '''
                the emit_output in this case tells TF how future emits look
                For the first call to loop_fn the emit_output corresponds to 
                the emit_structure which is then used to determine the size of 
                the zero_tensor for the emit_ta (defaults to cell.output_size). 
                '''
                emit_output = tf.tuple([tf.zeros([self.output_dim]), tf.zeros([self.output_dim]), 
                                        tf.zeros([self.output_dim])])  
                # tf.zeros([config.batch_size, output_dim], dtype=tf.float32)  # tf.zeros([output_dim]) 
                next_loop_state = output_ta
                '''
                this is the initial step, i.e. there is no output from a previous time step, what we feed here
                can highly depend on the data. In this case we just assign the actual input in the first time step.
                '''
                init_z = tf.zeros((self.batch_size, self.output_dim), dtype=tf.float32)
                #init_z = tf.random_normal((config.batch_size, output_dim), 0, 1, dtype=tf.float32)
                x_time = tf.layers.dropout(inputs_ta.read(time), rate= self.drop_rate_x)
                next_in = tf.concat([x_time, init_z],1)
            else:
                '''
                t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                here you can do whatever ou want with cell_output before assigning it to emit_output.
                In this case, we don't do anything pass the last state to the next
                '''
                
                next_cell_state = cell_state
                
                next_loop_state = self.get_next_loop_state(loop_state, cell_state, time)
                
        
                '''Next Output'''
                # cell_output = tf.Print(cell_output,[cell_output], message="cell_output")
                mean, var, current_z = self.get_output_step(cell_output)
    
                # current_z = tf.Print(current_z,[current_z], message="current z")    
                emit_output =  tf.tuple([mean, var, current_z])  
                # tf.tuple([mean, var])  tf.concat([mean, var],1)  cell_output mean
                
                next_in = tf.cond(finished,
                                  lambda: tf.zeros([self.batch_size, self.rnn_input_dim], dtype=tf.float32), 
                                  lambda: self.get_next_input(inputs_ta.read(time), current_z) )   
    
    
            next_input = tf.cond(finished, 
                                 lambda: tf.zeros([self.batch_size, self.rnn_input_dim], dtype=tf.float32), 
                                 lambda: next_in)
            next_input.set_shape([None, self.rnn_input_dim])
            
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)
        
        return loop_fn
        
'''
Inference Network for TVAE1
'''  

'''
Generator Network for TVAE
'''      

class RawRNNGener(BaseRawRNN):
    def __init__(self, cell_type, state_dim, input_, max_time, output_dim, reuse,
                 kinit=tf.contrib.layers.xavier_initializer(), 
                 bias_init=tf.constant_initializer(0.01), var_shared=False):
        super().__init__(input_, max_time, output_dim, cell_type, state_dim, reuse, kinit, bias_init)
        
        self.rnn_input_dim = self.output_dim

        self.act_out_mean = None
        self.act_out_var = tf.nn.softplus
        self.var_shared = var_shared
        
        self.is_sample = len(input_.get_shape().as_list())==2 
        self.is_time = not self.is_sample
        self.output_mean, self.output_var, self.output_z = self.my_build()
    
    def my_build(self):
        loop_fn, inputs_ta = self.get_loop_fn()
        output_list, state_list = self.build(loop_fn)
        outputs_mean = output_list[0]
        outputs_var = output_list[1]
        outputs_z = output_list[2]
        
        
        outputs_mean = get1toT(output_list[0], tf.zeros([self.batch_size, self.output_dim]), self.max_time) 
        outputs_var = get1toT(output_list[1], tf.ones([self.batch_size, self.output_dim]), self.max_time) 
            
        outputs_z = get1toT(output_list[2], self.input_, self.max_time)
        if(self.is_sample):
            outputs_z = get1toT(output_list[2], self.input_, self.max_time)
        else:
            outputs_z = get1toT(output_list[2], inputs_ta.read(0), self.max_time)
        
        states_all_c = state_list[0]
        states_all_h = state_list[1]
        
        print('Means: ', outputs_mean.get_shape().as_list())
        print('Vars: ', outputs_var.get_shape().as_list())
        print('Sampled z: ', outputs_z.get_shape().as_list())
        print('States c: ', states_all_c.get_shape().as_list())
        print('States h: ', states_all_h.get_shape().as_list())
        return outputs_mean, outputs_var, outputs_z
        
    
    def get_output_step(self, cell_output):
        with tf.variable_scope('mean', reuse=tf.AUTO_REUSE):
            mean_net = DenseNet(input_=cell_output,
                                hidden_dim=-1, 
                                output_dim=self.output_dim, 
                                num_layers=1, 
                                transfer_fct=None,
                                act_out=self.act_out_mean, 
                                reuse=tf.AUTO_REUSE, 
                                kinit=self.kinit,
                                bias_init=self.bias_init)
            
            mean = mean_net.output
            
        with tf.variable_scope('var', reuse=tf.AUTO_REUSE):
            if(self.var_shared):
                var = utils.get_variable(self.output_dim, 'var')
                var = tf.tile(var, [self.batch_size, 1])# [batch_size, var.dim]
            else:
                var_net = DenseNet(input_=cell_output,
                                    hidden_dim=-1, 
                                    output_dim=self.output_dim, 
                                    num_layers=1, 
                                    transfer_fct=None,
                                    act_out=self.act_out_var, 
                                    reuse=tf.AUTO_REUSE, 
                                    kinit=self.kinit,
                                    bias_init=self.bias_init)
                
                var = var_net.output

        eps = tf.random_normal((self.batch_size, self.output_dim), 0, 1, dtype=tf.float32)
        current_z = tf.add(mean, tf.multiply(tf.sqrt(var), eps))
        return mean, var, current_z
    
    def get_next_input(self, x_time, current_z):
       
        return 

    
    def get_loop_fn(self):

        inputs_ta, output_ta = self.get_tensor_arrays(self.input_)


        def loop_fn(time, cell_output, cell_state, loop_state):

            elements_finished = (time >= self.max_time)
            finished = tf.reduce_all(elements_finished)
            
            if cell_output is None:
                '''
                time == 0, used for initialization before first call to cell
                This is just to defined the desired shape of the tensors
                '''
                next_cell_state = self.cell.zero_state(self.batch_size, tf.float32)
                '''
                the emit_output in this case tells TF how future emits look
                For the first call to loop_fn the emit_output corresponds to 
                the emit_structure which is then used to determine the size of 
                the zero_tensor for the emit_ta (defaults to cell.output_size). 
                '''
                emit_output = tf.tuple([tf.zeros([self.output_dim]), tf.zeros([self.output_dim]), 
                                        tf.zeros([self.output_dim])])  
                # tf.zeros([config.batch_size, output_dim], dtype=tf.float32)  # tf.zeros([output_dim]) 
                next_loop_state = output_ta
                '''
                this is the initial step, i.e. there is no output from a previous time step, what we feed here
                can highly depend on the data. In this case we just assign the actual input in the first time step.
                '''
                
                if(self.is_sample):
                    next_in = self.input_
                else:
                    next_in = inputs_ta.read(time)
            else:
                '''
                t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                here you can do whatever ou want with cell_output before assigning it to emit_output.
                In this case, we don't do anything pass the last state to the next
                '''
                
                next_cell_state = cell_state
                
                next_loop_state = self.get_next_loop_state(loop_state, cell_state, time)
                
        
                '''Next Output'''
                # cell_output = tf.Print(cell_output,[cell_output], message="cell_output")
                mean, var, current_z = self.get_output_step(cell_output)
    
                # current_z = tf.Print(current_z,[current_z], message="current z")    
                emit_output =  tf.tuple([mean, var, current_z])  
                # tf.tuple([mean, var])  tf.concat([mean, var],1)  cell_output mean
                
                next_in = current_z
    
            if(self.is_sample):
                next_input = tf.cond(finished, 
                                     lambda: tf.zeros([self.batch_size, self.rnn_input_dim], dtype=tf.float32), 
                                     lambda: next_in)
            else:
                next_input = tf.cond(finished, 
                                     lambda: tf.zeros([self.batch_size, self.rnn_input_dim], dtype=tf.float32), 
                                     lambda: inputs_ta.read(time))
                
            next_input.set_shape([None, self.rnn_input_dim])
            
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)
        
        return loop_fn, inputs_ta
        
        
            