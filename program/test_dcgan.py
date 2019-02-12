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

def test_network_gan(test_size, latent_size, normalizarion, lr):
    print('Test Size',test_size)
    tf.reset_default_graph()
    weight_path = '../weigths/'+'DCGAN' + '.ckpt'
    z = tf.placeholder(tf.float32, [None, latent_size], name='latent')
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    generated = decoder(z,Training, 'g_','tanh')
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        y_values = []
        zs = []
        for i in range(test_size):
            sampled = sample_z(1, latent_size)
            feed = {z: sampled, Training:False}  
            output= sess.run([generated], feed_dict = feed)
            y_values +=[output[0][0,:,:,0]]
            zs +=[sampled[0]]
        y_values = np.array(y_values) 
        zs = np.array(zs)
        sess.close()
    #if (normalizarion):
    #    y_values = y_values*255.
    with open('../save/y_value_gan.pickle', 'wb') as f:
        pickle.dump(y_values, f)
    with open('../save/z_gan.pickle', 'wb') as f:
        pickle.dump(zs, f)
        
    print('Finished')

