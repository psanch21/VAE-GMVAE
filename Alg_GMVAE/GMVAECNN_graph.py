from GMVAE_graph import GMVAEGraph
import tensorflow as tf
import numpy as np

from networks.dense_net import DenseNet
from networks.conv_net import ConvNet3Gauss
from networks.deconv_net import DeconvNet3

'''
This is the Main TVAEGraph. Childs of this class are modifications.
'''
class GMVAECNNGraph(GMVAEGraph):
    def __init__(self, network_params,sigma=0.001, sigma_act=tf.nn.softplus,
                 transfer_fct= tf.nn.relu,learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),batch_size=32,
                 reuse=None, drop_rate=0.):
        
        super().__init__(network_params, sigma, sigma_act, transfer_fct, learning_rate, kinit, 
             batch_size, reuse, drop_rate)
        

    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()
    
    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.nchannel], name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch , [-1,self.x_flat_dim])
            self.beta = tf.placeholder(tf.float32,shape=(), name='beta')
            
            self.drop_rate = tf.placeholder(tf.float32,shape=(), name='drop_rate')
        
    def Pz_wy_graph(self, w_input, reuse):
        with tf.variable_scope('Pz_wy', reuse=reuse):
            
            Pz_wy = DenseNet(input_=w_input,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.hidden_dim, 
                            num_layers=1, 
                            transfer_fct=self.transfer_fct,
                            act_out=self.transfer_fct, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
            aux = Pz_wy.output
            
            z_wy_mean_list = list()
            with tf.variable_scope('mean', reuse=reuse):
                for i in range(self.K):
                    Pz_wy_mean = DenseNet(input_=aux,
                                          hidden_dim=self.hidden_dim, 
                                          output_dim=self.z_dim, 
                                          num_layers=0, 
                                          transfer_fct=self.transfer_fct,
                                          act_out=None, 
                                          reuse=reuse, 
                                          kinit=self.kinit,
                                          bias_init=tf.truncated_normal_initializer(stddev=1),
                                          drop_rate=self.drop_rate)
                    z_mean = Pz_wy_mean.dense(input_=aux, output_dim=self.z_dim, name='dense_' + str(i), act_func=None)
                    # z_mean = tf.scalar_mul(max_value,z_mean)
                    z_wy_mean_list.append(z_mean)
            
            z_wy_var_list = list()
            with tf.variable_scope('var', reuse=reuse):
                for i in range(self.K):
                    Pz_wy_var = DenseNet(input_=aux,
                                          hidden_dim=self.hidden_dim, 
                                          output_dim=self.z_dim, 
                                          num_layers=0, 
                                          transfer_fct=self.transfer_fct,
                                          act_out=self.sigma_act, 
                                          reuse=reuse, 
                                          kinit=self.kinit,
                                          bias_init=tf.truncated_normal_initializer(stddev=1),
                                          drop_rate=self.drop_rate)
                    z_var = Pz_wy_var.dense(input_=aux, output_dim=self.z_dim, name='dense_' + str(i), act_func=self.sigma_act)
                    # z_var = tf.scalar_mul(max_value,z_var)
                    z_wy_var_list.append(z_var)   
                
        z_wy_mean_stack = tf.stack(z_wy_mean_list) # [K, batch_size, z_dim]
        z_wy_var_stack = tf.stack(z_wy_var_list) # [K, batch_size, z_dim]
        z_wy_logvar_stack = tf.log(z_wy_var_stack)
        
        return z_wy_mean_list, z_wy_var_list
    
    def Px_z_graph(self, z_input, reuse):
        with tf.variable_scope('Pz_x', reuse=reuse):
            Px_z_mean = DeconvNet3(input_=z_input, 
                                   width=self.width, 
                                   height=self.height, 
                                   nchannels=self.nchannel, 
                                   reuse=reuse, 
                                   transfer_fct=self.transfer_fct,
                                   act_out=tf.nn.sigmoid, 
                                   drop_rate=self.drop_rate, 
                                   kinit=self.kinit,
                                   bias_init=self.bias_init)
            
        
            x_recons_mean_flat = Px_z_mean.output
            x_recons_mean_flat = tf.contrib.layers.flatten(x_recons_mean_flat)
        
        return x_recons_mean_flat
        
        
    def create_graph(self):
        print('\n[*] Defining Q(z|x)...')

        with tf.variable_scope('Qz_x', reuse=self.reuse):
            Qz_x = ConvNet3Gauss(input_=self.x_batch, 
                                 hidden_dim=self.z_dim*2,
                                 output_dim=self.z_dim, 
                                 reuse=self.reuse, 
                                 transfer_fct=self.transfer_fct,
                                 act_out_mean=None,
                                 act_out_var=tf.nn.softplus, 
                                 drop_rate=self.drop_rate, 
                                 kinit=self.kinit,
                                 bias_init=self.bias_init)
            

        
            self.z_x_mean = Qz_x.mean
            self.z_x_var = Qz_x.var
            
        
        print('\n[*] Reparameterization trick...')
        self.z_x_logvar = tf.log(self.z_x_var)
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z_x = tf.add(self.z_x_mean, tf.multiply(tf.sqrt(self.z_x_var), eps))

        print('\n[*] Defining Q(w|x)...')
        with tf.variable_scope('Qw_x', reuse=self.reuse):
            Qz_x = ConvNet3Gauss(input_=self.x_batch, 
                                 hidden_dim=self.w_dim*2,
                                 output_dim=self.w_dim, 
                                 reuse=self.reuse, 
                                 transfer_fct=self.transfer_fct,
                                 act_out_mean=None,
                                 act_out_var=tf.nn.softplus, 
                                 drop_rate=self.drop_rate, 
                                 kinit=self.kinit,
                                 bias_init=self.bias_init)
            

        
            self.w_x_mean = Qz_x.mean
            self.w_x_var = Qz_x.var

        
        print('\n[*] Reparameterization trick...')
        self.w_x_logvar = tf.log(self.w_x_var)
        eps = tf.random_normal((self.batch_size, self.w_dim), 0, 1, dtype=tf.float32)
        self.w_x = tf.add(self.w_x_mean, tf.multiply(tf.sqrt(self.w_x_var), eps))
    
        print('\n[*] Defining P(y|w,z)...')
        with tf.variable_scope('Py_wz', reuse=self.reuse):
            zw = tf.concat([self.w_x, self.z_x],1, name='wz_concat')
            Py_wz = DenseNet(input_=zw,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.K, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=None, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
            self.py_wz_logit = Py_wz.output
            self.py_wz = tf.nn.softmax(self.py_wz_logit)
        # Add small constant to avoid tf.log(0)
        self.log_py_wz = tf.log(1e-10 + self.py_wz)
            
        print('\n[*] Defining P(z|w,y)...')
        z_wy_mean_list, z_wy_var_list = self.Pz_wy_graph(self.w_x, self.reuse)
                
        self.z_wy_mean_stack = tf.stack(z_wy_mean_list) # [K, batch_size, z_dim]
        self.z_wy_var_stack = tf.stack(z_wy_var_list) # [K, batch_size, z_dim]
        self.z_wy_logvar_stack = tf.log(self.z_wy_var_stack)
        
        print('\n[*] Defining P(x|z)...')
        self.x_recons_mean_flat = self.Px_z_graph(self.z_x, self.reuse)

        eps = tf.random_normal(tf.shape(self.x_recons_mean_flat), 0, 1, dtype=tf.float32)
        self.x_recons_flat = tf.add(self.x_recons_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
        self.x_recons = tf.reshape(self.x_recons_flat , [-1,self.width, self.height, self.nchannel])

        
        print('\n[*] Defining sampling...')
        self.w_sample = tf.random_normal((self.batch_size, self.w_dim), 0, 1, dtype=tf.float32)
        self.z_wy_mean_list_sample, self.z_wy_var_list_sample = self.Pz_wy_graph(self.w_sample, True)
        self.z_sample_list = list()
        for i in range(self.K):
            eps = tf.random_normal(tf.shape(self.z_wy_mean_list_sample[i]), 0, 1, dtype=tf.float32)
            z_sample = tf.add(self.z_wy_mean_list_sample[i], tf.multiply(tf.sqrt(self.z_wy_var_list_sample[i]), eps))
            self.z_sample_list.append(z_sample)
            
        self.x_sample_mean_flat_list = list()
        self.x_sample_flat_list = list()
        self.x_sample_list = list()
        for i in range(self.K):
            x_sample_mean_flat = self.Px_z_graph(self.z_sample_list[i], True)
            self.x_sample_mean_flat_list.append(x_sample_mean_flat)
            
            eps = tf.random_normal(tf.shape(x_sample_mean_flat), 0, 1, dtype=tf.float32)
            x_sample_flat = tf.add(x_sample_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
            x_sample = tf.reshape(x_sample_flat , [-1,self.width, self.height, self.nchannel])
            
            self.x_sample_flat_list.append(x_sample_flat)
            self.x_sample_list.append(x_sample)
            
        

    '''  ------------------------------------------------------------------------------
                                     EVALUATE TENSORS
    ------------------------------------------------------------------------------ '''

     
        
        

