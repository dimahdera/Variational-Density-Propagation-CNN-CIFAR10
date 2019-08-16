## Copyright (C) <2018>  <Dimah Dera>
## Updated 2019
##
## Reference:
## Dimah Dera, Ghulam Rasool, Nidhal Bouaynaya, “Extended Variational Inference for Propagating
## Uncertainty in Convolutional Neural Networks”, IEEE International Workshop on Machine Learning
## for Signal Processing, October 2019.


import tensorflow as tf
import pickle
import sys
import os
import time
import numpy as np
import glob
import cv2
from numpy import linalg as LA
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import timeit
from scipy.misc import toimage




def activation(mu_g , mu_z, sigma_z, num_filters):
    
    activation_gradiant = tf.gradients(mu_g, mu_z)[0] # shape =[1, image_size,image_size,num_filters]
    gradient_matrix = tf.reshape(activation_gradiant,[1, -1, num_filters])# shape =[1, image_size*image_size, num_filters]    
    gradient_matrix=tf.expand_dims(gradient_matrix, 3)
    grad1 = tf.transpose(gradient_matrix, [0,2,1,3])
    grad_square = tf.squeeze(tf.matmul(grad1, grad1, transpose_b=True))# shape =[num_filters, image_size*image_size, image_size*image_size]

    #grad2 = tf.transpose(grad1,[0,1,3,2])
    #grad_square = tf.matmul(grad1,grad2) # shape =[1,num_filters,  image_size*image_size, image_size*image_size]
    #grad_square = tf.squeeze(grad_square)# shape =[num_filters, image_size*image_size, image_size*image_size]
    sigma_g = tf.multiply(sigma_z, grad_square)# shape =[num_filters, image_size*image_size, image_size*image_size]
    return sigma_g
    ######################################################

def UT_activation(mu_z, sigma_z, image_size, num_filters): 

    n_g = image_size * image_size  
    sigma_z_diag = tf.matrix_diag_part(sigma_z )#shape=[num_filter[0], image_size*image_size]
    L_gg = tf.sqrt(tf.clip_by_value(sigma_z_diag, 1e-15, 1e+15)   )
    L_g = tf.matrix_diag(L_gg) 

    L = np.sqrt(n_g)* L_g 
    x_hat1 = tf.transpose(tf.reshape(tf.squeeze(mu_z), [-1, num_filters]))#shape=[num_filter[0], image_size*image_size]
    #x_hat = tf.expand_dims(x_hat1, 1)
    #x_hat = tf.tile(x_hat, [1, image_size*image_size, 1])#shape=[num_filter[0], image_size*image_size, image_size*image_size]
    
    x_hat = tf.ones([1,1, image_size*image_size]) * tf.expand_dims(x_hat1, axis=-1)   

    sigma_points1 = x_hat  + L 
    sigma_points2 = x_hat  - L
    #mu_g1 = (1/(2*n_g))*(tf.reduce_sum(tf.nn.elu(sigma_points1) + tf.nn.elu(sigma_points2), 1)) #+ (3 - n_g)*tf.nn.elu(x_hat1) ) #shape=[num_filter[0], image_size*image_size]
    mu_g1 = (1/(2*n_g))*(tf.reduce_sum(tf.nn.elu(sigma_points1) + tf.nn.elu(sigma_points2), 2)) #shape=[num_filter[0], image_size*image_size]
    
    mu_g = tf.reshape(tf.transpose(mu_g1), [ image_size, image_size,num_filters ]) #shape=[ image_size, image_size,num_filters[0] ]
    mu_g = tf.expand_dims(mu_g, 0) #shape=[1,image_size, image_size,num_filters[0] ]
    
    #mu_g2 = tf.expand_dims(mu_g1, 1)
    #mu_g2 = tf.tile(mu_g2, [1, image_size*image_size,1])#shape=[num_filter[0], image_size*image_size, image_size*image_size]

    mu_g2 = tf.ones([1,1, image_size*image_size]) * tf.expand_dims(mu_g1, axis=-1)  

    
    P_ga1 = tf.nn.elu(sigma_points1) - mu_g2 #tf.nn.elu(x_hat) #shape=[num_filter[0], image_size*image_size,image_size*image_size]
    #P_gb1 = tf.matmul(P_ga1, tf.transpose(P_ga1, [0, 2, 1])) #shape=[num_filter[0], image_size*image_size, image_size*image_size] 
    #P_gb1 = tf.matmul(P_ga1, P_ga1,  transpose_a=True)  
    P_gb1 = tf.matmul(P_ga1, P_ga1,  transpose_b=True)   
  
    P_ga2 = tf.nn.elu(sigma_points2) - mu_g2 #tf.nn.elu(x_hat) #shape=[num_filter[0], image_size*image_size,image_size*image_size]
    #P_gb2 = tf.matmul(P_ga2, tf.transpose(P_ga2, [0, 2, 1])) #shape=[num_filter[0], image_size*image_size, image_size*image_size] 
    #P_gb2 = tf.matmul(P_ga2, P_ga2, transpose_a=True)
    P_gb2 = tf.matmul(P_ga2, P_ga2, transpose_b=True)
    
    P_gg = (1/(2*n_g))*(P_gb1 + P_gb2)#shape=[num_filter[0], image_size*image_size, image_size*image_size]
    return mu_g, P_gg

    
def UT_softmax(mu_f_fc1, sigma_f, num_labels):

    n_f = num_labels    
    sigma_f_diag = tf.diag_part(sigma_f)#shape=[ num_labels]  
    L_ff = tf.diag(tf.sqrt( tf.clip_by_value(sigma_f_diag, 1e-15, 1e+15)   ))#shape=[num_labels,num_labels]
    
    L_f = np.sqrt(n_f)* L_ff       
    fx_hat = tf.tile(mu_f_fc1, [ num_labels, 1])#shape=[num_labels,num_labels]
    
    fsigma_points1 = fx_hat  + L_f 
    fsigma_points2 = fx_hat  - L_f
    
    mu_y1 = (1/(2*n_f))*(tf.reduce_sum(tf.nn.softmax(fsigma_points1) + tf.nn.softmax(fsigma_points2), 0) )#+ (3 - n_f)*tf.nn.softmax(fx_hat1) ) #shape=[num_labels]    
    mu_y = tf.expand_dims(mu_y1, 0) #shape=[1, num_labels]
    mu_y2 = tf.tile(mu_y, [ num_labels,1])#shape=[num_labels,num_labels]
    P_ya1 = tf.nn.softmax(fsigma_points1) - mu_y2 #tf.nn.softmax(fx_hat,0) #shape=[num_labels, num_labels]  
    #P_yb1 = tf.matmul(P_ya1, tf.transpose(P_ya1)) #shape=[num_labels, num_labels]  
    P_yb1 = tf.matmul(P_ya1, P_ya1, transpose_a=True) 
    
    P_ya2 = tf.nn.softmax(fsigma_points2) - mu_y2 #tf.nn.softmax(fx_hat,0) #shape=[num_labels, num_labels]   
    #P_yb2 = tf.matmul(P_ya2, tf.transpose(P_ya2)) #shape=[num_labels,num_labels] 
    P_yb2 = tf.matmul(P_ya2, P_ya2, transpose_a=True)
      
    sigma_y = (1/(2*n_f))*(P_yb1 + P_yb2)#shape=[num_labels, num_labels]
    return mu_y, sigma_y


    


    
    

def max_pooling(sigma_g, argmax, num_filters, new_size, image_size ):
         
    argmax1= tf.transpose(argmax, [0, 3, 1, 2])
    argmax2 = tf.reshape(argmax1,[1, num_filters, -1])#shape=[1, num_filters, new_size*new_size]
    
    new_sigma_g =  tf.reshape(sigma_g,[ num_filters*image_size*image_size,-1])     
    x_index = tf.mod(tf.floor_div(argmax2,tf.constant(num_filters ,shape=[1,num_filters, new_size*new_size], dtype='int64')),tf.constant(image_size ,shape=[1,num_filters, new_size*new_size], dtype='int64')) 
    
    aux = tf.floor_div(tf.floor_div(argmax2,tf.constant(num_filters,shape=[1,num_filters, new_size*new_size], dtype='int64')),tf.constant(image_size,shape=[1,num_filters, new_size*new_size], dtype='int64'))    
    y_index = tf.mod(aux,tf.constant(image_size,shape=[1,num_filters,new_size*new_size], dtype='int64'))
    index = tf.multiply(y_index,image_size) + x_index
    index = tf.squeeze(index) # shape=[num_filters,new_size*new_size]      
    for i in range(num_filters):
        if(i==0):
            ind1 = tf.gather(index, tf.constant(i))
            new_ind = ind1
        else:
            ind1 = (image_size*image_size*i)+ tf.gather(index, tf.constant(i))
            new_ind = tf.concat([new_ind,ind1],0) # shape=[num_filters*new_size*new_size] 
    column1 = tf.gather(new_sigma_g,new_ind) 
    column2 = tf.reshape(column1, [num_filters, new_size*new_size, -1])
    column3 = tf.transpose(column2, [0, 2, 1]) 
    column4 = tf.reshape(column3, [num_filters*image_size*image_size, -1])
    final = tf.gather(column4,new_ind)
    sigma_p = tf.reshape(final,[num_filters,new_size*new_size,new_size*new_size]) #shape=[num_filters,new_size*new_size, new_size*new_size] 
    return sigma_p


def fully_connected(fc_weight_mu ,fc_weight_sigma, mu_b, sigma_p, num_filters, new_size, num_labels ):

    fc_weight_mu1 = tf.reshape(fc_weight_mu, [new_size*new_size, num_filters,num_labels]) #shape=[num_filters,new_size*new_size,num_labels]
    fc_weight_mu1T = tf.transpose(fc_weight_mu1,[1,2,0]) #shape=[num_filters,num_labels,new_size*new_size]
    
    muhT_sigmab = tf.matmul(fc_weight_mu1T, sigma_p)#shape=[num_filters,num_labels,new_size*new_size]
    muhT_sigmab_mu = tf.matmul(muhT_sigmab, tf.transpose(fc_weight_mu1,[1,0,2]))#shape=[num_filters,num_labels,num_labels]
    muhT_sigmab_mu = tf.reduce_sum(muhT_sigmab_mu, 0) #shape=[num_labels,num_labels]

    diag_elements = tf.matrix_diag_part(sigma_p) #shape=[num_filters, new_size*new_size]     
    diag_sigma_b =tf.reshape(diag_elements,[-1]) #shape=[new_size*new_size*num_filters]
    
    tr_sigma_b = tf.reduce_sum(diag_sigma_b)#shape=[1]
    mu_bT_mu_b = tf.reduce_sum(tf.multiply(mu_b, mu_b),1)  
    mu_bT_mu_b = tf.squeeze(mu_bT_mu_b)#shape=[1]    
    tr_sigma_h_sigma_b = tf.multiply(tf.log(1. + tf.exp(fc_weight_sigma)), tr_sigma_b) # shape=[num_labels] 
    mu_bT_sigma_h_mu_b = tf.multiply(tf.log(1. + tf.exp(fc_weight_sigma)), mu_bT_mu_b) # shape=[num_labels]     
    tr_sigma_h_sigma_b = tf.diag(tr_sigma_h_sigma_b) #shape=[num_labels,num_labels]
    mu_bT_sigma_h_mu_b = tf.diag(mu_bT_sigma_h_mu_b) #shape=[num_labels,num_labels]    
    sigma_f = tr_sigma_h_sigma_b + muhT_sigmab_mu + mu_bT_sigma_h_mu_b #shape=[num_labels,num_labels]

    return sigma_f

def first_convolution(x ,conv1_weight_sigma , num_filters, patch_size,num_channel, pad="VALID" ):


    x_train_patches = tf.extract_image_patches(x, ksizes=[1, patch_size, patch_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding = pad)# shape=[1, image_size, image_size, patch_size*patch_size*num_channel]
    x_train_matrix = tf.reshape(x_train_patches,[1, -1, patch_size*patch_size*num_channel])# shape=[1, image_size*image_size, patch_size*patch_size*num_channel]
    
    X_XTranspose = tf.matmul(x_train_matrix, x_train_matrix, transpose_b=True)# shape=[image_size*image_size, image_size*image_size ] dimension of vectorized slice in the tensor z
    X_XTranspose = tf.squeeze(tf.ones([1,1,1, num_filters]) * tf.expand_dims(X_XTranspose, axis=-1))


    #X_XTranspose = tf.matmul(x_train_matrix, tf.transpose(x_train_matrix, [0, 2, 1]))# shape=[1, image_size*image_size, image_size*image_size ] dimension of vectorized slice in the tensor z
    #X_XTranspose=tf.expand_dims(X_XTranspose, 1)
    #X_XTranspose = tf.tile(X_XTranspose, [1, num_filters,1,1])#shape=[1, num_filter, image_size*image_size, image_size*image_size]
    #X_XTranspose = tf.transpose(X_XTranspose, [0, 2, 3, 1])#shape=[1,image_size*image_size, image_size*image_size, num_filter]
    #X_XTranspose = tf.squeeze(X_XTranspose) #shape=[image_size*image_size, image_size*image_size, num_filter]
    sigma_z = tf.multiply(tf.log(1. + tf.exp(conv1_weight_sigma)), X_XTranspose)#shape=[image_size*image_size, image_size*image_size, num_filter]
    sigma_z = tf.transpose(sigma_z, [2,0,1])#shape=[num_filter,image_size*image_size, image_size*image_size]  
    return sigma_z


def intermediate_convolution(w_mean, w_s, mu_z, pre_sigma, patch_size , num_filters0, num_filters1, im_size, new_im_size, pad="VALID"):
    new_im_si = new_im_size*new_im_size*new_im_size*new_im_size
    new_im_si1 = new_im_size*new_im_size
    im_si = im_size*im_size
    pa_si = patch_size*patch_size
    
    mu_cov = tf.transpose(tf.reshape(w_mean, [pa_si,num_filters0, num_filters1]), [2,1,0])#[num_filters1,num_filters0,pa_si]
    mu_conv = tf.expand_dims(mu_cov, 2)
    mu_conv = tf.tile(mu_conv, [1,1,new_im_si,1])
    mu_conv = tf.expand_dims(mu_conv, 3)#[num_filters1,num_filters0,new_im_si,1, pa_si]
    mu_convT = tf.transpose(mu_conv, [0,1,2,4,3]) #[num_filters1,num_filters0,new_im_si, pa_si, 1]
    
    pre_sigma1 = tf.expand_dims(tf.reshape(pre_sigma,[num_filters0*im_si,im_size,im_size]),3)
    patch_sig = tf.extract_image_patches(pre_sigma1 ,ksizes=[1,patch_size,patch_size,1],strides=[1,1,1,1],rates=[1,1,1,1],padding=pad) #[num_filters0*im_si,new_im_size,new_im_size,pa_si]
    patch_sig1 = tf.transpose(tf.reshape(patch_sig, [num_filters0,im_size*im_size, -1]), [0,2,1])
    patch_sig2 = tf.expand_dims(tf.reshape(patch_sig1, [-1, im_size,im_size]),3)
    patch_sig2 = tf.extract_image_patches(patch_sig2 ,ksizes=[1,patch_size,patch_size,1],strides=[1,1,1,1],rates=[1,1,1,1],padding=pad)#[num_filters0*new_im_si1*pa_si, new_im_size,new_im_size,pa_si ]
    patch_sig3 = tf.transpose(tf.reshape(patch_sig2, [num_filters0, new_im_si1,pa_si,new_im_si1,pa_si ]), [0, 1,3,2,4])
    patch_sig_f = tf.reshape(patch_sig3, [num_filters0,new_im_si,pa_si,pa_si])
    mod_sig = tf.expand_dims(patch_sig_f, 0)
    mod_sig = tf.tile(mod_sig, [num_filters1, 1, 1, 1, 1]) #[num_filters1,num_filters0,new_im_si,pa_si, pa_si]

    mu_h_sigma = tf.squeeze(tf.matmul(tf.matmul(mu_conv, mod_sig),mu_convT))#[num_filters1,num_filters0,new_im_si]
    mu_h_sigma_mu_h = tf.reshape(tf.reduce_sum(mu_h_sigma, 1), [num_filters1,new_im_si1,new_im_si1 ])#[num_filters1,new_im_si1,new_im_si1]

    diag_sigma = tf.matrix_diag_part(pre_sigma)#shape=[num_filters0,im_size*im_size] 
    diag_sigma = tf.transpose(diag_sigma) #shape=[im_size*im_size,num_filters0] 
    diag_sigma = tf.expand_dims(tf.reshape(diag_sigma, [im_size, im_size,num_filters0] ),0)#shape=[1, im_size,im_size,num_filters0]     
    diag_sigma_patches=tf.squeeze(tf.extract_image_patches(diag_sigma,ksizes=[1,patch_size,patch_size,1],strides=[1,1,1,1],rates=[1,1,1,1],padding=pad))# shape=[new_im_size,new_im_size, patch_size*patch_size*num_filters0]

          
    trace = tf.expand_dims(tf.reshape(tf.reduce_sum(diag_sigma_patches,2), [-1]), 1)# shape=[ new_im_size* new_im_size,1]
    trace = tf.tile(trace, [1,num_filters1])#shape=[ new_im_size*new_im_size, num_filters1]      
    trace = tf.transpose(tf.multiply( tf.log(1. + tf.exp(w_s)), trace ))#shape=[num_filters1, new_im_size*new_im_size]    
    trace1 = tf.matrix_diag(trace) #shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]
    
    
    mu_z = tf.reshape(tf.squeeze(mu_z), [-1, num_filters1]) # shape=[ new_im_size*new_im_size,num_filters1]
    mu_z_square = tf.multiply(mu_z ,mu_z) # shape=[ new_im_size*new_im_size,num_filters1]
    mu_z_square = tf.transpose(tf.multiply( tf.log(1. + tf.exp(w_s)), mu_z_square )) # shape=[num_filters1, new_im_size*new_im_size]
    mu_z_square1 = tf.matrix_diag(mu_z_square) #shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]
    
    sigma_z = trace1 + mu_h_sigma_mu_h + mu_z_square1 #shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]

    return sigma_z



def intermediate_convolution_approx(w_mean, w_s, mu_g, pre_sigma, patch_size , num_filters0, num_filters1, im_size, new_im_size, pad="VALID"):

    new_im_si = new_im_size*new_im_size*new_im_size*new_im_size
    new_im_si1 = new_im_size*new_im_size
    im_si = im_size*im_size
    pa_si = patch_size*patch_size
    

    diag_sigma = tf.matrix_diag_part(pre_sigma)#shape=[num_filters0,im_size*im_size] 
    diag_sigma = tf.transpose(diag_sigma) #shape=[im_size*im_size,num_filters0] 
    diag_sigma = tf.expand_dims(tf.reshape(diag_sigma, [im_size, im_size,num_filters0] ),0)#shape=[1, im_size,im_size,num_filters0]     
    diag_sigma_patches=tf.squeeze(tf.extract_image_patches(diag_sigma,ksizes=[1,patch_size,patch_size,1],strides=[1,1,1,1],rates=[1,1,1,1],padding=pad))
    # shape=[new_im_size,new_im_size, patch_size*patch_size*num_filters0]

    diag_sigma_g = tf.transpose(tf.reshape(diag_sigma_patches, [-1, patch_size*patch_size*num_filters0] ))
    # shape=[ patch_size*patch_size*num_filters0,   new_im_size*new_im_size]
    mu_cov = tf.transpose( tf.reshape(w_mean, [patch_size*patch_size*num_filters0, num_filters1]))
    # shape[num_filters1 , patch_size*patch_size*num_filters0]

    
    mu_cov_squar = tf.multiply(mu_cov, mu_cov)# shape[num_filters1,patch_size*patch_size*num_filters0]    
    mu_wT_sigmags_mu_w1 = tf.matmul(mu_cov_squar, diag_sigma_g)#shape=[num_filters1,new_im_size*new_im_size]
    
    mu_wT_sigmags_mu_w = tf.matrix_diag(mu_wT_sigmags_mu_w1) #shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]


          
    trace = tf.reshape(tf.reduce_sum(diag_sigma_patches,2), [-1])# shape=[ new_im_size* new_im_size]
    #trace = tf.tile(trace, [1,num_filters1])#shape=[ new_im_size*new_im_size, num_filters1]
    trace = tf.ones([1, num_filters1]) * tf.expand_dims(trace, axis=-1)#shape=[ new_im_size*new_im_size, num_filters1]
    
    trace = tf.transpose(tf.multiply( tf.log(1. + tf.exp(w_s)), trace ))#shape=[num_filters1, new_im_size*new_im_size]    
    trace1 = tf.matrix_diag(trace) #shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]

     # mu_g , shape=[1, im_size, im_size,num_filters0]
    mu_g_patches = tf.reshape(tf.squeeze(tf.extract_image_patches(mu_g,ksizes=[1,patch_size,patch_size,1],strides=[1,1,1,1],rates=[1,1,1,1],padding=pad)), [-1,patch_size*patch_size*num_filters0 ])
    # shape=[new_im_size*new_im_size, patch_size*patch_size*num_filters0]
    mu_gT_mu_g = tf.matmul(mu_g_patches, mu_g_patches, transpose_b=True)# shape=[new_im_size*new_im_size,new_im_size*new_im_size]    
       
    mu_gT_mu_g1 = tf.ones([1,1, num_filters1]) * tf.expand_dims(mu_gT_mu_g, axis=-1)
    # shape=[new_im_size*new_im_size, new_im_size*new_im_size, num_filters1]    
    sigmaw_mu_gT_mu_g = tf.transpose(tf.multiply( tf.log(1. + tf.exp(w_s)), mu_gT_mu_g1 ))
    # shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]
    

    sigma_z = trace1 + mu_wT_sigmags_mu_w + sigmaw_mu_gT_mu_g #shape=[num_filters1, new_im_size*new_im_size, new_im_size*new_im_size]

    return sigma_z
    
    




