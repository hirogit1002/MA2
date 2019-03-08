import tensorflow as tf
import tensorboard as tb 
import numpy as np
from PIL import Image, ImageFilter
#import cv2
import pandas as pd
import os
import sys
import time
import pickle
from layerfunctions import*
from model import*
from imgproc import*


def test_network(data, latent_size, normalizarion, model_name,lr):
    models = {'AE':AE,'VAE':VAE_test}
    data = np.array(data)
    n = len(data)
    weight_path = '../weigths/'+model_name +'_'+str(latent_size)+ '.ckpt'
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    keep_prob = tf.placeholder(tf.float32)
    Batch_size = tf.placeholder(tf.int32)
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    out, cost_val, optimizer, fv = models[model_name](x, keep_prob, 1, latent_size, Training,lr)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        y_values = []
        zs = []
        for i in data:
            imgs = np.array(Image.open(i).convert('L'))
            if (normalizarion):
                imgs = norm_intg(imgs[np.newaxis,:])
            else:
                imgs = imgs[np.newaxis,:,:,np.newaxis].astype(np.float32)
                
            print('Loss:', cost_val.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:1, Training:False}))
            y_value = out.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:1, Training:False})
            z = fv.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:1, Training:False})
            y_values +=[y_value[0,:,:,0]]
            zs +=[z[0]]
        y_values = np.array(y_values) 
        zs = np.array(zs)
        sess.close()
    if (normalizarion):
        y_values = y_values*255.
        
    path_y_value = '../save/y_value_'+model_name+'.pickle'
    path_z = '../save/z_'+model_name+'.pickle'
    with open(path_y_value, 'wb') as f:
        pickle.dump(y_values, f)
    with open(path_z , 'wb') as f:
        pickle.dump(zs, f)

