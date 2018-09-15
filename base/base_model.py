import os
import tensorflow as tf


class BaseModel:
    def __init__(self, checkpoint_dir, summary_dir, result_dir):
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir
        self.result_dir = result_dir
        
        # init the global step
        # self.init_global_step()
        # init the epoch counter
        # self.init_cur_epoch()
        


    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, saver, global_step_tensor):
        print("Saving model...")
        saver.save(sess, self.checkpoint_dir + '/', global_step_tensor)
        print("Model saved" )

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, saver):
        retval = False
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
            retval = True
        else:
            print("Model does NOT exist")
        return retval
    
    '''
    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    '''
    def train_epoch(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError

