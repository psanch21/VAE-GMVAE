import json
from bunch import Bunch
import os
import argparse
import utils.constants as const 


def parser_basic():
    desc = "Tensorflow implementation of VAE and GMVAE"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=20,  help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of inputs used for each iteration')
    # reconstrucion variance
    parser.add_argument('--sigma', type=float, default=0.1, help='Parameter that defines the variance of the output Gaussian distribution')
    parser.add_argument('--l_rate', type=float, default=1e-3, help='Parameter of the optimization function')

    parser.add_argument('--z_dim', type=int, default=3, help='Dimension of the latent variable z')
    parser.add_argument('--w_dim', type=int, default=3, help='Dimension of the latent variable w. Only for GMVAE')
    parser.add_argument('--K_clusters', type=int, default=10, help='Number of modes of the latent variable z. Only for GMVAE')
    
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter in the KL')
    parser.add_argument('--drop_prob', type=float, default=0, help='Dropout regularizer parameter')
    
    # hidden units in the NNs
    parser.add_argument('--hidden_dim', type=int, default=50, help='Number of neurons of each dense layer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of dense layers in each network')

    parser.add_argument('--checkpoint_dir', type=str, default='saved_models', help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--logdir', type=str, default='logdir', help='Directory name to save training logs')
    parser.add_argument('--dataset_name', type=str, default='MNIST or FREY')

    parser.add_argument('--train', type=int, default=1, help='Flag to set train')
    parser.add_argument('--summary', type=int, default=1,help='Flag to set TensorBoard summary')
    parser.add_argument('--restore', type=int, default=0, help='Flag to restore model')
    parser.add_argument('--plot', type=int, default=0, help='Flag to set plot')
    parser.add_argument('--results', type=int, default=0, help='Flag to get results')
    parser.add_argument('--early_stopping', type=int, default=1, help='Set to 1 to early_stopping')


    parser.add_argument('--model_type', type=int, default=0, help='Fixes the model and architecture')
    parser.add_argument('--extra', type=str, default='', help='Extra name to identify the model')

    # TODO: Que pasa con esto?
    parser.add_argument('--max_mean', type=int, default=10)
    parser.add_argument('--max_var', type=int, default=1)
    # parser.add_argument('--save_file', type=str, default='1-VAE-model-MNIST')

    return parser

def check_args(args):
    '''
    This method check the values provided are correct
    '''
    try:
        assert args.epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --z_dim
    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')

    return args

def get_args():
    parser = parser_basic()
    args = parser.parse_args()
    return check_args(args)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    # config = Bunch(config_dict)

    return config, config_dict


def process_args(args,model):
    config = Bunch(args)
    
    config.model_name = model + '_'+\
                        config.dataset_name+ '_' + \
                        str(config.sigma).replace('.', '')+ '_'+\
                        str(config.z_dim) + '_'+ \
                        str(config.hidden_dim)  + '_'+\
                        str(config.num_layers)
    
    if(config.extra is not ''):
        config.model_name += '_' + config.extra
    config.summary_dir = os.path.join("experiments/summary/", config.model_name)
    config.checkpoint_dir = os.path.join("experiments/checkpoint/", config.model_name)
    config.results_dir = os.path.join("experiments/results/", config.model_name)
    
    flags = Bunch()
    flags.train = args['train']
    flags.restore = args['restore']
    flags.results = args['results']
    flags.plot = args['plot']
    flags.early_stopping =  args['early_stopping']
    
    return config, flags

def get_config_and_flags(args):
    aux_name = ''
    if(args.model_type==const.VAE):
        aux_name+='V'
    if(args.model_type==const.VAECNN):
        aux_name+='VC'

    # aux_name+= '_'+str(int(args.beta))
    
    return process_args(vars(args), aux_name)  
