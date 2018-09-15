import tensorflow as tf

import os, sys

# os.chdir(os.path.dirname(os.getcwd()))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../') # Consider also the parent directory for the module lookup


from utils.dataset import Dataset
from bunch import Bunch
from utils.args_processing import get_args, get_config_and_flags
from utils.logger import Logger
from utils.utils import save_img

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utils
import utils.constants as const 
plt.close("all")
'''  ------------------------------------------------------------------------------
                                     EXAMPLE OF USE
    ------------------------------------------------------------------------------ '''

# VAE_main.py --sigma=0.001 --z_dim=10 --hidden_dim=128 --num_layers=2 --extra=_test --epochs=50 --batch_size=128 --dataset_name=MNIST --beta=1 --model_type=0 --drop_prob=0.3 --l_rate=0.05 --train=1 --results=1 --plot=1 --restore=1 --early_stopping=1
'''  ------------------------------------------------------------------------------
                                     GET ARGUMENTS
    ------------------------------------------------------------------------------ '''
# capture the config path from the run arguments
# then process the json configuration file
args = get_args()
config, flags = get_config_and_flags(args)

# create the experiments dirs
utils.create_dirs([config.summary_dir, config.checkpoint_dir, config.results_dir])
utils.save_args(args, config.summary_dir)


'''  ------------------------------------------------------------------------------
                                     GET DATA
    ------------------------------------------------------------------------------ '''
print('\n Loading data...')
data_train, data_valid, data_test = utils.load_data(config.dataset_name)


'''  ------------------------------------------------------------------------------
                                     GET NETWORK PARAMS
    ------------------------------------------------------------------------------ '''
network_params = Bunch()
network_params.input_height = data_train.height
network_params.input_width = data_train.width
network_params.input_nchannels = data_train.num_channels

network_params.hidden_dim =  config.hidden_dim
network_params.z_dim =  config.z_dim
network_params.num_layers =  config.num_layers

'''  ------------------------------------------------------------------------------
                                     MODEL OPERATIONS
    ------------------------------------------------------------------------------ '''
from VAE_model import VAEModel
vae_model = VAEModel(network_params,sigma=config.sigma, sigma_act=utils.softplus_bias,
                       transfer_fct=tf.nn.relu,learning_rate=config.l_rate,
                       kinit=tf.contrib.layers.xavier_initializer(),
                       batch_size=config.batch_size, drop_rate=config.drop_prob, 
                       epochs=config.epochs, checkpoint_dir=config.checkpoint_dir, 
                       summary_dir=config.summary_dir, result_dir=config.results_dir, 
                       restore=flags.restore, model_type=config.model_type)
print('\nNumber of trainable paramters', vae_model.trainable_count)
if(flags.train==1):
    vae_model.train(data_train, data_valid, enable_es=flags.early_stopping)
    

'''  ------------------------------------------------------------------------------
                                      RESULTS
    ------------------------------------------------------------------------------ '''
if(flags.results==1):
    from VAE_visualize import VAEVisualize
    visualize = VAEVisualize(config.model_name, config.results_dir, (10, 15))
       
    x_samples, z_samples = vae_model.generate_samples(data_test)
    
    visualize.samples(x_samples, z_samples)
    
    x_input, x_labels, x_recons, z_recons = vae_model.reconstruct_input(data_test)
    
    visualize.recons(x_input,x_labels,  x_recons, z_recons)
    

if(config.plot==1):
    plt.show()
else:
    plt.close("all")

