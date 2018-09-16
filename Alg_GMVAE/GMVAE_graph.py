from base.base_graph import BaseGraph
import tensorflow as tf
import numpy as np

from networks.dense_net import DenseNet

'''
This is the Main TVAEGraph. Childs of this class are modifications.
'''
class GMVAEGraph(BaseGraph):
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
        self.w_dim = network_params.w_dim    
        self.K = network_params.K    
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
            with tf.variable_scope('mean'):
                for i in range(self.K):
                    Pz_wy_mean = DenseNet(input_=aux,
                                          hidden_dim=self.hidden_dim, 
                                          output_dim=self.z_dim, 
                                          num_layers=0, 
                                          transfer_fct=self.transfer_fct,
                                          act_out=None, 
                                          reuse=self.reuse, 
                                          kinit=self.kinit,
                                          bias_init=tf.truncated_normal_initializer(stddev=1),
                                          drop_rate=self.drop_rate)
                    z_mean = Pz_wy_mean.dense(input_=aux, output_dim=self.z_dim, name='dense_' + str(i), act_func=None)
                    # z_mean = tf.scalar_mul(max_value,z_mean)
                    z_wy_mean_list.append(z_mean)
            
            z_wy_var_list = list()
            with tf.variable_scope('var'):
                for i in range(self.K):
                    Pz_wy_var = DenseNet(input_=aux,
                                          hidden_dim=self.hidden_dim, 
                                          output_dim=self.z_dim, 
                                          num_layers=0, 
                                          transfer_fct=self.transfer_fct,
                                          act_out=self.sigma_act, 
                                          reuse=self.reuse, 
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
            Px_z_mean = DenseNet(input_=z_input,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.x_flat_dim, 
                            num_layers=2, 
                            transfer_fct=self.transfer_fct,
                            act_out=tf.nn.sigmoid, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            x_recons_mean_flat = Px_z_mean.output
        
        return x_recons_mean_flat
        
        
    def create_graph(self):
        print('\n[*] Defining Q(z|x)...')

        with tf.variable_scope('Qz_x_mean', reuse=self.reuse):
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
        
            self.z_x_mean = Qz_x_mean.output
            
        with tf.variable_scope('Qz_x_var', reuse=self.reuse):
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
        
            self.z_x_var = Qz_x_var.output
        
        print('\n[*] Reparameterization trick...')
        self.z_x_logvar = tf.log(self.z_x_var)
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z_x = tf.add(self.z_x_mean, tf.multiply(tf.sqrt(self.z_x_var), eps))

        print('\n[*] Defining Q(w|x)...')

        with tf.variable_scope('Qw_x_mean', reuse=self.reuse):
            Qw_x_mean = DenseNet(input_=self.x_batch_flat,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.w_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=None, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            self.w_x_mean = Qw_x_mean.output
            
        with tf.variable_scope('Qw_x_var', reuse=self.reuse):
            Qw_x_var = DenseNet(input_=self.x_batch_flat,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.w_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=self.sigma_act, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.drop_rate)
        
            self.w_x_var = Qw_x_var.output
        
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
        
        self.x_sample_mean_flat_list = list()
        self.x_sample_flat_list = list()
        self.x_sample_list = list()
        for z_wy_mean_sample in self.z_wy_mean_list_sample:
            x_sample_mean_flat = self.Px_z_graph(z_wy_mean_sample, True)
            self.x_sample_mean_flat_list.append(x_sample_mean_flat)
            
            eps = tf.random_normal(tf.shape(x_sample_mean_flat), 0, 1, dtype=tf.float32)
            x_sample_flat = tf.add(x_sample_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
            x_sample = tf.reshape(x_sample_flat , [-1,self.width, self.height, self.nchannel])
            
            self.x_sample_flat_list.append(x_sample_flat)
            self.x_sample_list.append(x_sample)
            
        
        
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.name_scope('reconstruct'):
            
            self.reconstruction = -0.5 / self.sigma * tf.reduce_sum(tf.square(self.x_recons_flat - self.x_batch_flat), 1)
            
        self.loss_reconstruction_m = -tf.reduce_mean(self.reconstruction)

        # Regularization Term:
        with tf.name_scope('cond_prior'):
            # Shape  tf.reduce_sum(z_logvar,1): [batch_size]
            logq = -0.5 * tf.reduce_sum(self.z_x_logvar,1) - 0.5 * tf.reduce_sum(tf.divide(
                    tf.square(self.z_x - self.z_x_mean), self.z_x_var), 1)
            # Shape z_logvars_stack = tf.stack(z_logvars): [K, batch_size, z_dim]
            # Shape b = tf.reduce_sum(z_logvars_stack,2): [K, batch_size]
            # Shape log_det_sigma = tf.transpose(b) : [batch_size, K]
            # Shape y_logits: [batch_size, K]
            # Shape tf.reduce_sum(tf.multiply(y_logits,log_det_sigma),1) : [batch_size]
    
            # TODO: Que z usar?
            z_wy = tf.expand_dims(self.z_x, 2)
    
            z_wy = tf.tile( z_wy , [1,1, self.K])# [batch_size, z_dim, K]
            z_wy = tf.transpose(z_wy,perm=[2,0,1])  # [K, batch_size, z_dim]
            log_det_sigma =  tf.transpose(tf.reduce_sum(self.z_wy_logvar_stack,2)) # [batch_size, K ]
    
            # Shape a = tf.squared_difference(z_wy, z_means_stack): [K, batch_size, z_dim]
            # Shape b = tf.divide(a, tf.exp(z_logvars_stack)): [K, batch_size, z_dim]
            # Shape tf.reduce_sum (b, [0,2])  : [batch_size]
    
            aux = tf.divide(tf.square(z_wy - self.z_wy_mean_stack), self.z_wy_var_stack) # [K, batch_size, z_dim]
            aux = tf.reduce_sum(aux, 2) # [K, batch_size]
            aux = tf.transpose(aux) # [batch_size, K]
            aux = tf.multiply(self.py_wz, aux) # [batch_size, K]
            aux = tf.reduce_sum(aux,1) # [batch_size]
            logp = -0.5 * tf.reduce_sum(tf.multiply(self.py_wz, log_det_sigma),1) -0.5 * aux
            self.cond_prior = logq - logp
            
        self.cond_prior_m = tf.reduce_mean(self.cond_prior)
        
        # Regularization Term:
        with tf.name_scope('w_prior'):
            self.KL_w = 0.5 * tf.reduce_sum(self.w_x_var + tf.square(self.w_x_mean) - 1  - self.w_x_logvar, 1)
    
        self.KL_w_m = tf.reduce_mean(self.KL_w)
        
        # Regularization Term:
        with tf.name_scope('y_prior'):
            # Shape logy_logits : [batch_size,10]
            self.y_prior  = -np.log(self.K, dtype='float32')  -1/self.K * tf.reduce_sum(self.log_py_wz, axis=1)
        self.y_prior_m = tf.reduce_mean(self.y_prior)
    
        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.L2 = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        
        
        with tf.variable_scope("loss", reuse=self.reuse):
            self.loss = -tf.reduce_mean(self.reconstruction - self.cond_prior - self.KL_w - self.y_prior) # + self.regularization_cost # shape = [None,]
        
        with tf.variable_scope("optimizer" ,reuse=self.reuse):
        
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    '''  ------------------------------------------------------------------------------
                                     EVALUATE TENSORS
    ------------------------------------------------------------------------------ '''
    def partial_fit(self, session, x, beta=1, drop_rate=0.):
        tensors = [self.train_step, self.loss, self.loss_reconstruction_m, self.cond_prior_m, 
                   self.KL_w_m, self.y_prior_m, self.L2]
        feed_dict = {self.x_batch: x, self.beta: beta, self.drop_rate: drop_rate }
        _, loss, recons, cond_prior, KL_w, y_prior, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, cond_prior, KL_w, y_prior, L2_loss
    
    def evaluate(self, session, x, beta=1, drop_rate=0.):
        tensors = [self.loss, self.loss_reconstruction_m, self.cond_prior_m, self.KL_w_m, self.y_prior_m, self.L2]
        feed_dict = {self.x_batch: x, self.beta: beta, self.drop_rate: drop_rate }
        loss, recons, cond_prior, KL_w, y_prior, L2_loss = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, cond_prior, KL_w, y_prior, L2_loss

    def get_z_matrix(self, session, x, beta=1):
        feed_dict = {self.x_batch: x, self.beta: beta, self.drop_rate: 0. }
        return session.run(self.z,feed_dict=feed_dict)
    
    def generate_samples(self, session, x, beta=1, num_batches=5):
        feed = {self.x_batch: np.zeros(x.shape), self.beta: beta, self.drop_rate: 0. }
        w = []
        z = []
        
        tensors =  [self.w_sample]
        for i in range(num_batches):
            w_tmp = session.run(tensors, feed_dict = feed)
            w.extend(w_tmp)
    
        w = np.array(w)
        return w
    
    def reconstruct_input(self, session, x, beta=1):

        feed = {self.x_batch: x, self.beta: beta, self.drop_rate: 0. }
        
        tensors = [self.x_recons,self.z_x, self.w_x, self.py_wz]

        return session.run( tensors, feed_dict = feed)
     
        
        

