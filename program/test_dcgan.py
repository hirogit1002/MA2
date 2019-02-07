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

def sample_z(batch_size, latent_size):
    return np.random.uniform(-1., 1., size=[batch_size, latent_size])

def test_network_gan(test_size, latent_size, normalizarion, lr):
    tf.reset_default_graph()
    perm = np.random.permutation(len(data))
    data = np.array(data[perm])
    train_data = np.array(data[:-test_size])
    test_data = np.array(data[-test_size:])
    weight_path = '../weigths/'+'DCGAN' + '.ckpt'
    n_test = test_size
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    z = tf.placeholder(tf.float32, [None, latent_size], name='latent')
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    generated = decoder(z,Training)
    sig, D_logits = discriminator(x,Training, reuse=False)
    sig_, D_logits_ = discriminator(generated, Training, reuse=True)
    loss_real, loss_fake, g_loss, val_loss_real, val_loss_fake, val_g_loss = loss_gan(D_logits,D_logits_)                    
    d_loss = loss_real + loss_fake
    val_d_loss = val_loss_real + val_loss_fake

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        y_values = []
        zs = []
        for i in data:
            imgs = np.array(Image.open(i).convert('L'))
            imgs = imgs[np.newaxis,:,:,np.newaxis]
            if (normalizarion):
                imgs = imgs/255.
            
            sampled = sample_z(1, latent_size)
            feed = {z: sampled, x: test_imgs, Training:False}  
            output, test_d_cost, test_g_cost = sess.run([generated], feed_dict = feed)
            print(output.shape)
            y_values +=[output[0,:,:,0]]
            zs +=[sampled[0]]
        y_values = np.array(y_values) 
        zs = np.array(zs)
        sess.close()
    if (normalizarion):
        y_values = y_values*255.
    with open('../save/y_value_gan.pickle', 'wb') as f:
        pickle.dump(y_values, f)
    with open('../save/z_gan.pickle', 'wb') as f:
        pickle.dump(zs, f)

