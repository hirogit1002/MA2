import tensorflow as tf
import tensorboard as tb 
import numpy as np
import cv2
import pandas as pd
import os
import sys
import time
import pickle
from layerfunctions import*
from model import*
from imgproc import*


def test_network(data, latent_size, normalizarion =True,shp=[-1, 64, 64, 1], model_name='AE'):
    models = {'AE':AE,'VAE':VAE_test}
    data = np.array(data)
    data_shape = shp
    data_shape[0] = None
    n = len(data)
    weight_path = '../weigths/'+model_name + '.ckpt'
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, data_shape, name='InputData')
    keep_prob = tf.placeholder(tf.float32)
    Batch_size = tf.placeholder(tf.int32)
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    out, cost_trn, cost_val, optimizer, fv = models[model_name](x, keep_prob, Batch_size, latent_size, Training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        y_values = []
        zs = []
        for i in data:
            imgs = np.array([cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY)])
            imgs = imgs[:,:,:,np.newaxis]
            if (normalizarion):
                imgs = imgs/255.
            print('Cost:', cost_val.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:n, Training:False}))
            y_value = out.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:n, Training:False})
            z = fv.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:n, Training:False})
            y_values +=[y_value]
            zs +=[z]
        sess.close()
    if (normalizarion):
        y_values = y_values*255
    with open('../save/y_value.pickle', 'wb') as f:
        pickle.dump(y_values, f)
    with open('../save/z.pickle', 'wb') as f:
        pickle.dump(zs, f)

