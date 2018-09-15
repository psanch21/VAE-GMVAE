#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:19:22 2018

@author: psanch
"""

import tensorflow as tf

class BaseGraph:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)
        
    def build_graph(self):
        raise NotImplementedError