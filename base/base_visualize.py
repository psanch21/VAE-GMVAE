#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:19:22 2018

@author: psanch
"""

import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utils

class BaseVisualize:
    def __init__(self, model_name, result_dir, fig_fize):
        self.model_name = model_name
        self.result_dir = result_dir
        self.fig_size = fig_fize
        self.colors = {0:'black', 1:'grey', 2:'blue', 3:'cyan', 4:'lime', 5:'green', 6:'yellow', 7:'gold', 8:'red', 9:'maroon'}
        
    def save_img(self, fig, name):
        utils.save_img(fig, self.model_name, name, self.result_dir)
        return
        
    def reduce_dimensionality(self, var, perplexity=10):
        dim = var.shape[-1]
        if(dim>2):
            tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=1000)
            var_2d = tsne.fit_transform(var)
        else:
            var_2d = np.asarray(var)
        return var
        
    def scatter_variable(self, var, labels, title, perplexity=10):
        f, axarr = plt.subplots(1, 1, figsize=self.fig_size)
        var_2d = self.reduce_dimensionality(var)
        for number, color in self.colors.items():
            axarr.scatter(x=var_2d[labels==number, 0], y=var_2d[labels==number, 1], color=color, label=str(number))
    
    
        axarr.legend()
        axarr.grid()
        f.suptitle(title, fontsize=20)
        return f