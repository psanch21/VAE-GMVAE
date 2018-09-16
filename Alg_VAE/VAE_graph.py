from base.base_graph import BaseGraph
import tensorflow as tf
import numpy as np

from networks.dense_net import DenseNet

'''
This is the Main TVAEGraph. Childs of this class are modifications.
'''
class VAEGraph(BaseGraph):
    def __init__(self, network_params,sigma=0.001, sigma_act=tf.nn.softplus,
                 transfer_fct= tf.nn.relu,learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),batch_size=32,
                 reuse=None, drop_rate=0.):
        
        super().__init__(learning_rate)
        
        self.width = network_params['input_width']
        self.height = network_params['input_height']
        self.nchannel = network_params['input_nchannels']
        
        self.hidden_dim = network_params['hidden_dim']
        self.z_dim = network_params.z_dim    
        self.num_layers = network_params['num_layers'] # Num of Layers in P(x|z)
        self.sigma = sigma # Sigma in P(x|z)
        
        self.sigma_act = sigma_act # Actfunc for NN modeling variance
        
        self.x_flat_dim = self.width * self.height * self.nchannel
        
        self.transfer_fct = transfer_fct
        self.kinit = kinit
        self.bias_init = tf.constant_initializer(0.)
        self.batch_size = batch_size
        
        self.reuse = reuse
        
        

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
        
        
    def create_graph(self):
        print('\n[*] Defining encoder...')

        with tf.variable_scope('encoder_mean', reuse=self.reuse):
            Qz_x_mean = DenseNet(input_=self.x_batch_flat,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.z_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=None, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            self.encoder_mean = Qz_x_mean.output
            
        with tf.variable_scope('encoder_var', reuse=self.reuse):
            Qz_x_var = DenseNet(input_=self.x_batch_flat,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.z_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=tf.nn.softplus, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            self.encoder_var = Qz_x_var.output
        
        print('\n[*] Reparameterization trick...')
        self.encoder_logvar = tf.log(self.encoder_var)
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.encoder_mean, tf.multiply(tf.sqrt(self.encoder_var), eps))
       
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder_mean', reuse=self.reuse):
            Px_z_mean = DenseNet(input_=self.z,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.x_flat_dim, 
                            num_layers=2, 
                            transfer_fct=self.transfer_fct,
                            act_out=tf.nn.sigmoid, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            self.decoder_mean_flat = Px_z_mean.output
        
        eps = tf.random_normal(tf.shape(self.decoder_mean_flat), 0, 1, dtype=tf.float32)
        self.decoder_x_flat = tf.add(self.decoder_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
        self.decoder_x = tf.reshape(self.decoder_x_flat , [-1,self.width, self.height, self.nchannel])

        
        print('\n[*] Defining sampling...')
        self.z_sample = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        
        with tf.variable_scope('decoder_mean', reuse=True):
            Px_z_mean = DenseNet(input_=self.z_sample,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.x_flat_dim, 
                            num_layers=2, 
                            transfer_fct=self.transfer_fct,
                            act_out=tf.nn.sigmoid, 
                            reuse=True, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            self.samples_mean_flat = Px_z_mean.output
        
        eps = tf.random_normal(tf.shape(self.samples_mean_flat), 0, 1, dtype=tf.float32)
        self.samples_flat = tf.add(self.samples_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
        self.samples = tf.reshape(self.samples_flat , [-1,self.width, self.height, self.nchannel])
        
        
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.name_scope('reconstruct'):
            
            self.reconstruction = -0.5 / self.sigma * tf.reduce_sum(tf.square(self.decoder_mean_flat - self.x_batch_flat), 1)
            
        self.loss_reconstruction_m = -tf.reduce_mean(self.reconstruction)

        # Regularization Term:
        with tf.name_scope('KL_divergence'):
            self.KL = 0.5 * tf.reduce_sum(self.encoder_var + tf.square(self.encoder_mean) - 1 - self.encoder_logvar, 1)
            
        self.KL_m = tf.reduce_mean(self.KL)

        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope("loss", reuse=self.reuse):
            self.loss = -tf.reduce_mean(self.reconstruction - self.beta*self.KL) # + self.regularization_cost # shape = [None,]
        
        with tf.variable_scope("optimizer" ,reuse=self.reuse):
        
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    '''  ------------------------------------------------------------------------------
                                     EVALUATE TENSORS
    ------------------------------------------------------------------------------ '''
    def partial_fit(self, session, x, beta=1, drop_rate=0.):
        tensors = [self.train_step, self.loss, self.loss_reconstruction_m, self.KL_m, self.regularization_cost]
        feed_dict = {self.x_batch: x, self.beta: beta, self.drop_rate: drop_rate }
        _, loss, recons, KL, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, KL, L2_loss
    
    def evaluate(self, session, x, beta=1, drop_rate=0.):
        tensors = [self.loss, self.loss_reconstruction_m, self.KL_m, self.regularization_cost]
        feed_dict = {self.x_batch: x, self.beta: beta, self.drop_rate: drop_rate }
        loss, recons, KL, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, KL, L2_loss

    def get_z_matrix(self, session, x, beta=1):
        feed_dict = {self.x_batch: x, self.beta: beta, self.drop_rate: 0. }
        return session.run(self.z,feed_dict=feed_dict)
    
    def generate_samples(self, session, x, beta=1):
        feed = {self.x_batch: np.zeros(x.shape), self.beta: beta, self.drop_rate: 0. }
        samples = []
        z = []
        
        tensors =  [self.samples,  self.z_sample]
        for i in range(5):
            s, z_tmp = session.run(tensors, feed_dict = feed)
            samples.extend(s)
            z.extend(z_tmp)
        
        samples = np.array(samples)
        z = np.array(z)
        return samples, z
    
    def reconstruct_input(self, session, x, beta=1):

        feed = {self.x_batch: x, self.beta: beta, self.drop_rate: 0. }
        
        tensors= [self.decoder_x,self.z]

        return session.run( tensors, feed_dict = feed)
     
        
        

