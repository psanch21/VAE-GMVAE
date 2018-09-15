from VAE_graph import VAEGraph
import tensorflow as tf
import numpy as np

from networks.dense_net import DenseNet
from networks.conv_net import ConvNet3Gauss
from networks.deconv_net import DeconvNet3

'''
This is the Main TVAEGraph. Childs of this class are modifications.
'''
class VAECNNGraph(VAEGraph):
    def __init__(self, network_params,sigma=0.001, sigma_act=tf.nn.softplus,
                 transfer_fct= tf.nn.relu,learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),batch_size=32,
                 reuse=None, drop_rate=0.):
        
        super().__init__(network_params, sigma, sigma_act, transfer_fct, learning_rate, kinit, 
             batch_size, reuse, drop_rate)
        
        
        

    def build_graph(self):
        self.create_inputs()
        self.create_TVAE()
        self.create_loss_optimizer()
    
    def create_TVAE(self):
        print('\n[*] Defining encoder...')

        with tf.variable_scope('encoder', reuse=self.reuse):
            
            Qz_x = ConvNet3Gauss(input_=self.x_batch, 
                                 hidden_dim=self.z_dim*2,
                                 output_dim=self.z_dim, 
                                 reuse=self.reuse, 
                                 transfer_fct=self.transfer_fct,
                                 act_out_mean=None,
                                 act_out_var=tf.nn.softplus, 
                                 drop_rate=self.drop_rate, 
                                 kinit=tf.contrib.layers.xavier_initializer(),
                                 bias_init=tf.constant_initializer(0.0))
            
        
            self.encoder_mean = Qz_x.mean
            self.encoder_var = Qz_x.var
        
        print('\n[*] Reparameterization trick...')
        self.encoder_logvar = tf.log(self.encoder_var)
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.encoder_mean, tf.multiply(tf.sqrt(self.encoder_var), eps))
       
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder_mean', reuse=self.reuse):
            
            Px_z_mean = DeconvNet3(input_=self.z, 
                                   width=self.width, 
                                   height=self.height, 
                                   nchannels=self.nchannel, 
                                   reuse=self.reuse, 
                                   transfer_fct=self.transfer_fct,
                                   act_out=tf.nn.sigmoid, 
                                   drop_rate=self.drop_rate, 
                                   kinit=self.kinit,
                                   bias_init=self.bias_init)

        
            self.decoder_mean = Px_z_mean.output
        
        
        self.decoder_mean_flat = tf.contrib.layers.flatten(self.decoder_mean)

        eps = tf.random_normal(tf.shape(self.decoder_mean_flat), 0, 1, dtype=tf.float32)
        self.decoder_x_flat = tf.add(self.decoder_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
        self.decoder_x = tf.reshape(self.decoder_x_flat , [-1,self.width, self.height, self.nchannel])

        
        print('\n[*] Defining sampling...')
        self.z_sample = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        
        with tf.variable_scope('decoder_mean', reuse=True):
            
            Px_z_mean = DeconvNet3(input_=self.z_sample, 
                                   width=self.width, 
                                   height=self.height, 
                                   nchannels=self.nchannel, 
                                   reuse=True, 
                                   transfer_fct=self.transfer_fct,
                                   act_out=tf.nn.sigmoid, 
                                   drop_rate=self.drop_rate, 
                                   kinit=self.kinit,
                                   bias_init=self.bias_init)
            
        
            self.samples_mean = Px_z_mean.output
        self.samples_mean_flat = tf.contrib.layers.flatten(self.samples_mean)
        eps = tf.random_normal(tf.shape(self.samples_mean_flat), 0, 1, dtype=tf.float32)
        self.samples_flat = tf.add(self.samples_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
        self.samples = tf.reshape(self.samples_flat , [-1,self.width, self.height, self.nchannel])
        
        

     
        
        

