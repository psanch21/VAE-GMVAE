import tensorflow as tf
import model1
import model2
import model3
import model4_bias
import model5



def VAE_graph(model_name, shape, reuse,rate, args):
    '''
    model1: VAE with Dense layers
    model2: VAE with Convolutional layers
    model3: GMVAE with just one latent variable z, Dense layers
    model4_bias:  GMVAE Dense layers
    model5: GMVAE Convolutional layers
    '''
    if(model_name == 'model1'):
        # shape = [batch_size, data_dim]
        return model1.graph(shape, reuse, rate, args)

    if(model_name == 'model2'):
        # shape = [batch_size,input_w, input_h, num_channels]
        return model2.graph(shape, reuse, rate, args)

    if(model_name == 'model3'):
        return model3.graph(shape, reuse, rate, args)

    if(model_name == 'model4_bias'):
        return model4_bias.graph(shape, reuse, rate, args)

    if(model_name == 'model5'):
        return model5.graph(shape, reuse, rate, args)


    return None
