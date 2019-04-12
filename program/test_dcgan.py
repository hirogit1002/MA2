import tensorflow as tf
import tensorboard as tb 
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import os
import sys
import time
import pickle
from layerfunctions import*
from model import*
from imgproc import*
from extractor_dcgan import*

def test_network_gan(test_size, latent_size, normalizarion, lr, vector_path, pool_typ):
    print('Test Size',test_size)
    tf.reset_default_graph()
    weight_path = '../weigths/'+'DCGAN'+'_'+str(latent_size) + '.ckpt'
    z = tf.placeholder(tf.float32, [None, latent_size], name='latent')
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    generated = generator(z, Training,1)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        y_values = []
        zs = []
        if(len(vector_path)>0):
            with open(vector_path, 'rb') as f:
                samples = pickle.load(f)
                test_size = len(samples)
        else:
            samples = sample_z(test_size, latent_size)
        for i in range(test_size):
            sampled = samples[i]
            feed = {z: sampled, Training:False}  
            output= sess.run([generated], feed_dict = feed)
            y_values +=[output[0][0,:,:,0]]
            zs +=[sampled[0]]
        y_values = np.array(y_values) 
        zs = np.array(zs)
        sess.close()
    print("Save represented data")
    with open(('../save/y_value_gan_'+str(latent_size)+'.pickle'), 'wb') as f:
        pickle.dump(y_values, f)
    print("Save latent variable")
    with open(('../save/z_gan_'+str(latent_size)+'.pickle'), 'wb') as f:
        pickle.dump(zs, f)
    print("Extract feature vectors")
    extractor(latent_size,pool_typ ,Reuse=False)
    print('Finished')

