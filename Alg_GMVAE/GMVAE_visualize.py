#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:24:14 2018

@author: psanch
"""
from base.base_visualize import BaseVisualize
import numpy as np
import matplotlib.pyplot as plt


class GMVAEVisualize(BaseVisualize):
    def __init__(self, model_name, result_dir, fig_size):
        super().__init__(model_name, result_dir,fig_size )

    
    def samples(self, x_samples, z_samples, num_samples_to_plot=5):
        f, axarr = plt.subplots(num_samples_to_plot, num_samples_to_plot, figsize=self.fig_size)
        
        count = 0
        idx_vec = np.random.randint(0,x_samples.shape[0])
        for i in range(num_samples_to_plot):
            for j in range(num_samples_to_plot):
                axarr[i, j].imshow(x_samples[count, :].reshape([x_samples.shape[1], x_samples.shape[2]]), cmap='gray')
                axarr[i, j].axis('off')
                count+=1
                     
        # axarr[0,0].set_title('z')
        
        # f.legend()
        st = f.suptitle('Images Generated')
        f.tight_layout()
        st.set_y(0.98)
        f.subplots_adjust(top=0.90)
        self.save_img(f, 'data_generation')
        
        return 
    
    def recons(self, x_input, x_labels, x_recons, z_recons, w_recons, y_recons, num_samples_to_plot=5):

        f, axarr = plt.subplots(num_samples_to_plot, 3, figsize=self.fig_size)
        
        x = np.arange(y_recons.shape[-1])
        for i in range(num_samples_to_plot):
            axarr[i, 0].imshow(x_input[i, :].reshape([x_input.shape[1], x_input.shape[2]]), cmap='gray')
            axarr[i, 1].imshow(x_recons[i, :].reshape([x_input.shape[1], x_input.shape[2]]), cmap='gray')
            axarr[i, 2].bar(x, y_recons[i,:])
            
        # f.legend()
        
        
        st = f.suptitle('Image Reconstruction')
        f.tight_layout()
        st.set_y(0.98)
        f.subplots_adjust(top=0.90)
        self.save_img(f, 'data_recons')

        # f, axarr = plt.subplots(1, 2, figsize=self.fig_size)
        f = self.scatter_variable(z_recons, x_labels, 'Scatter Plot of z')
        self.save_img(f, 'scatter_z')
        
        f = self.scatter_variable(w_recons, x_labels, 'Scatter Plot of w')
        self.save_img(f, 'scatter_w')
        return      
  