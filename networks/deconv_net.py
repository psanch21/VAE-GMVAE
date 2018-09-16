#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:32:57 2018

@author: pablosanchez
"""
import tensorflow as tf
import utils.constants as const
from networks.dense_net import DenseNet

class DeconvNet(object):
    def __init__(self,  width, height, nchannels, reuse, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0., kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0)):
        
        self.width = width
        self.height = height
        self.nchannels = nchannels

        self.transfer_fct = transfer_fct
        self.act_out = act_out
        self.reuse = reuse
        self.drop_rate = drop_rate
        
        self.kinit= kinit
        self.bias_init = bias_init


   

    def build(self, input_):
        raise NotImplementedError

        

 
    def deconv_layer(self, input_, filters, k_size, stride, padding, name, act_func=tf.nn.relu):
        deconv = tf.layers.conv2d_transpose(input_, 
                                            filters,
                                            k_size, 
                                            strides=stride, 
                                            padding=padding, 
                                            activation=act_func, 
                                            kernel_initializer=self.kinit, 
                                            bias_initializer=self.bias_init, 
                                            name=name, 
                                            reuse=self.reuse)
        print('[*] Layer (',deconv.name, ') output shape:', deconv.get_shape().as_list())
    
        return deconv



class DeconvNet3(DeconvNet):
    def __init__(self, input_, width, height, nchannels, reuse, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0., kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0)):
        super().__init__(width, height, nchannels,reuse, transfer_fct, act_out, drop_rate, kinit, bias_init)
        
        self.output = self.build(input_)


   
    
    def build(self, input_):
        aux_size = self.width//2//2
        aux_size_2 = self.height//2//2
        initial_n_channels = 64
        out_dense_dim = aux_size*aux_size_2*initial_n_channels
        hidden_dim = input_.get_shape()[-1].value*3
        
        dense = DenseNet(input_=input_,
                         hidden_dim=hidden_dim, 
                         output_dim=out_dense_dim, 
                         num_layers=2, 
                         transfer_fct=self.transfer_fct,
                         act_out=self.transfer_fct, 
                         reuse=self.reuse, 
                         kinit=self.kinit,
                         bias_init=self.bias_init,
                         drop_rate=self.drop_rate)
        x = dense.output
        x = tf.reshape(x, [-1,aux_size,aux_size_2,initial_n_channels])
        x = self.deconv_layer(input_=x, 
                              filters=64, 
                              k_size=4,  #[4, 4]
                              stride=2, 
                              padding='SAME', 
                              name='deconv_1',  
                              act_func=self.transfer_fct)
        
        x = self.deconv_layer(input_=x, 
                            filters=32, 
                            k_size=4,  #[4, 4]
                            stride=2, 
                            padding='SAME', 
                            name='deconv_2',  
                            act_func=self.transfer_fct)
        x = self.deconv_layer(input_=x, 
                            filters=self.nchannels, 
                            k_size=4,  #[4, 4]
                            stride=1, 
                            padding='SAME', 
                            name='deconv_3',  
                            act_func=self.act_out)
        
        return x
  