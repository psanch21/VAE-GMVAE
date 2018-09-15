#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:32:57 2018

@author: pablosanchez
"""
import tensorflow as tf
import utils.constants as const
class BaseRawRNN(object):
    def __init__(self,input_, max_time, output_dim, cell_type, state_dim, reuse,kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.01)):
        self.state_dim = state_dim
        self.reuse = reuse
        self.cell_type = cell_type
        self.cell = self.get_a_cell(cell_type)
        
        self.kinit = kinit
        self.bias_init = bias_init
        
        self.input_ = input_
        self.batch_size = input_.get_shape()[0].value
        self.max_time = max_time
        self.input_dim=input_.get_shape()[-1].value
        self.output_dim = output_dim
        
        
    def my_build(self):
        raise NotImplementedError
    
    def build(self, loop_fn):
        
        # outputs_ta: tuple [means, vars]
        # last_state:last state tuple [c, h]
        # states_all_ta: all states concat(c,h)
        outputs_ta, last_state, states_all_ta = tf.nn.raw_rnn(self.cell, loop_fn)
        output_list = self.get_outputs_rnn(outputs_ta)
        
        # states_all_c, states_all_h
        state_list = self.get_states_rnn(states_all_ta)
        
    
        return output_list , state_list
    
    def get_outputs_rnn(self, outputs_ta):
        output_list = list()
        for out in outputs_ta:
            out_tmp = tf.transpose(out.stack(), perm=[1, 0, 2])
            output_list.append(out_tmp)
        
        return output_list
        
    def get_states_rnn(self,states_all_ta):
        states_all = tf.transpose(states_all_ta.stack(), perm=[1, 0, 2])
        if(self.cell_type == const.LSTM_CELL):
            states_all_c, states_all_h = tf.split(states_all, num_or_size_splits=2, axis=2)
        elif(self.cell_type == const.RNN_CELL): # Both are the same
            states_all_c = states_all
            states_all_h = states_all
        return states_all_c, states_all_h
    
    
    def get_tensor_arrays(self, input_):
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_time, 
                                   clear_after_read=False, name='input_concat')
        if(len(input_.get_shape().as_list())==3 ):        
            inputs_ta = inputs_ta.unstack(tf.transpose(input_, perm=[1, 0, 2]))

        # store trained/sampled pixel
        output_ta = tf.TensorArray(size=self.max_time, dtype=tf.float32, name='output_concat')  
        
        return inputs_ta, output_ta
    
    def get_next_loop_state(self, loop_state, cell_state, time):
        if(self.cell_type == const.LSTM_CELL):
            next_loop_state  = loop_state.write(time - 1,tf.concat([cell_state[0], cell_state[1]],1))
        elif(self.cell_type == const.RNN_CELL):
            next_loop_state  = loop_state.write(time - 1, cell_state)
        
        return next_loop_state
    def get_loop_fn(self, inputs):

        
        def loop_fn(time, cell_output, cell_state, loop_state):
            """
            Loop function that allows to control input to the rnn cell and manipulate cell outputs.
            :param time: current time step
            :param cell_output: output from previous time step or None if time == 0
            :param cell_state: cell state from previous time step
            :param loop_state: custom loop state to share information between different iterations of 
            this loop fn
            
            :return: tuple consisting of
              finished: tensor of size [bach_size] which is True for sequences that have reached their 
              end, needed because of variable sequence size
              next_input: input to next time step
              next_cell_state: cell state forwarded to next time step
              emit_output: The first return argument of raw_rnn. This is not necessarily the output of 
              the RNN cell,but could e.g. be the output 
              of a dense layer attached to the rnn layer.
              next_loop_state: loop state forwarded to the next time step
            """
    
            elements_finished = (time >= max_time)
            finished = tf.reduce_all(elements_finished)
            
            
            if cell_output is None:
                '''
                time == 0, used for initialization before first call to cell
                This is just to defined the desired shape of the tensors
                '''
                next_cell_state = cell.zero_state(batch_size, tf.float32)
                '''
                the emit_output in this case tells TF how future emits look
                For the first call to loop_fn the emit_output corresponds to the emit_structure which is
                then used to determine the size of the zero_tensor for the emit_ta (defaults to cell.output_size). 
                '''
                emit_output = tf.tuple([tf.zeros([output_dim]), tf.zeros([output_dim]), tf.zeros([output_dim])])  
                # tf.zeros([config.batch_size, output_dim], dtype=tf.float32)  # tf.zeros([output_dim]) 
                next_loop_state = output_ta
                '''
                this is the initial step, i.e. there is no output from a previous time step, what we feed here
                can highly depend on the data. In this case we just assign the actual input in the first time step.
                '''
                init_z = tf.zeros((batch_size, output_dim), dtype=tf.float32)
                #init_z = tf.random_normal((config.batch_size, output_dim), 0, 1, dtype=tf.float32)
                x_time = tf.layers.dropout(inputs_ta.read(time), rate=rate_x)
                next_in = tf.concat([x_time, init_z],1)
            else:
                '''
                t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                here you can do whatever ou want with cell_output before assigning it to emit_output.
                In this case, we don't do anything
                pass the last state to the next
                '''
                
                # next_cell_state = cell_state

                
                # emit_output =  tf.tuple([mean, var, current_z])  
                
    
                # next_in = tf.cond(finished,lambda: tf.zeros([batch_size, rnn_input_dim], dtype=tf.float32), 
                
                # next_loop_state  = loop_state.write(time - 1,tf.concat([cell_state[0], cell_state[1]],1))


                
            
    
                
            # next_input = tf.cond(finished, lambda: tf.zeros([batch_size, rnn_input_dim], dtype=tf.float32), lambda: next_in)
            # next_input.set_shape([None, rnn_input_dim])
            
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)
        
        return loop_fn
    
    def get_output_step(self, cell_output):
        raise NotImplementedError
        
    def get_next_input(self, *arg):
        raise NotImplementedError
        
        
    
    def get_a_cell(self, cell_type=const.LSTM_CELL):
        out = None
    
        if(cell_type == const.LSTM_CELL):
            out = tf.nn.rnn_cell.BasicLSTMCell(self.state_dim, reuse=self.reuse)
        elif(cell_type == const.RNN_CELL):
            out = tf.contrib.rnn.BasicRNNCell(num_units=self.state_dim, reuse=self.reuse)  
        # out = tf.nn.rnn_cell.DropoutWrapper(out, output_keep_prob=1 - drop_prob, state_keep_prob=1 - drop_prob)
        return out
    