import tensorflow as tf
import tensorboard as tb 
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from layerfunctions import*
from model_ae import*
from imgproc import*


def test_network(data, latent_size, normalizarion =True,shp=[-1, 64, 64, 1], model_name='AE'):
    models = {'AE':AE}
    data = np.array(data)
    data_shape = shp
    data_shape[0] = None
    n = len(data)
    weight_path = '../../weigths/'+model_name + '.ckpt'
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
        imgs = np.array([cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY) for i in data])
        imgs = imgs[:,:,:,np.newaxis]
        if (normalizarion):
            imgs = imgs/255.
        print('Cost:', cost_val.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:n, Training:False}))
        y_value = out.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:n, Training:False})
        z = fv.eval(feed_dict={x: imgs, keep_prob:1., Batch_size:n, Training:False})
        sess.close()
    if (normalizarion):
        y_value = y_value*255
    return y_value, z
