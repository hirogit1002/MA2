import tensorflow as tf
import tensorboard as tb 
import numpy as np
from PIL import Image, ImageFilter
#import cv2
import pandas as pd
import os
import sys
import time
from layerfunctions import*
from model import*
from imgproc import*

    
def train_network_gan(data, test_size, batch_size,init,latent_size, normalizarion, epochs, logs_path, lr):
    tf.reset_default_graph()
    perm = np.random.permutation(len(data))
    data = np.array(data[perm])
    train_data = np.array(data[:-test_size])
    test_data = np.array(data[-test_size:])
    weight_path = '../weigths/'+'DCGAN' + '.ckpt'
    n = len(train_data)
    n_test = len(test_data)
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    z = tf.placeholder(tf.float32, [None, latent_size], name='latent')
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    generated = decoder(z,Training, 'g_','tanh')
    sig, D_logits = discriminator(x,Training, reuse=False)
    sig_, D_logits_ = discriminator(generated, Training, reuse=True)
    loss_real, loss_fake, g_loss, val_loss_real, val_loss_fake, val_g_loss = loss_gan(D_logits,D_logits_)                    
    d_loss = loss_real + loss_fake
    val_d_loss = val_loss_real + val_loss_fake

    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    
    # Optimizer
    dis_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_vars)
    gen_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_vars)
    
    with tf.name_scope('training'):
        tf.summary.scalar("g_loss", g_loss)
        tf.summary.scalar("d_loss", d_loss)
    with tf.name_scope('validation'):
        tf.summary.scalar("g_loss", val_g_loss)
        tf.summary.scalar("d_loss", val_d_loss)

    trn_summary = tf.summary.merge_all(scope='training')
    val_summary = tf.summary.merge_all(scope='validation')
        
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: #allow_soft_placement=True, 
        if tf.gfile.Exists(logs_path):
            tf.gfile.DeleteRecursively(logs_path)
        file_writer = tf.summary.FileWriter(logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        if (init==False):
            saver.restore(sess, weight_path)
        for epoch in range(epochs):
            sum_d_loss = 0
            sum_g_loss = 0
            n_batches = int(n / batch_size)
            # Loop over all batches
            counter = 0
            train_data = np.random.permutation(train_data)
            for i in range(n_batches):
                batch_x = train_data[i*batch_size:(i+1)*batch_size]
                imgs = np.array([np.array(Image.open(i).convert('L')) for i in batch_x])
                if (normalizarion):
                    imgs = norm_intg(imgs,'tanh')
                else:
                    imgs = imgs[:,:,:,np.newaxis]
                # Run optimization op (backprop) and cost op (to get loss value)
                feed = {z: sample_z(batch_size, latent_size), x: imgs, Training:True}      
                _,train_d_loss = sess.run([dis_op, d_loss], feed_dict=feed )
                _,train_g_loss = sess.run([gen_op, g_loss], feed_dict=feed )
                _,res_trn ,train_cost = sess.run([gen_op, trn_summary, g_loss], feed_dict=feed)
                sum_d_loss += (train_d_loss / n_batches)
                sum_g_loss += (train_g_loss/ n_batches)
                sys.stdout.write("\r%s" % "batch: {}/{}, d_loss: {}, g_loss: {}".format(counter+1, np.int(n/batch_size)+1, sum_d_loss/(i+1), sum_g_loss/(i+1)))
                sys.stdout.flush()
                counter +=1
            file_writer.add_summary(res_trn, (epoch+1))
            saver.save(sess, weight_path)

            # Display logs per epoch step
            print('')
            print('Epoch', epoch+1, ' / ', epochs, 'D Cost:', sum_d_loss/epochs, 'G Cost:', sum_g_loss/epochs)
            print('')
            test_imgs = np.array([np.array(Image.open(i).convert('L')) for i in test_data])
            if (normalizarion):
                test_imgs = norm_intg(test_imgs,'tanh')
            else:
                test_imgs =  test_imgs[:,:,:,np.newaxis]                
            feed = {z: sample_z(n_test, latent_size), x: test_imgs, Training:False}  
            test_d_cost, test_g_cost = sess.run([val_d_loss,val_g_loss], feed_dict = feed)
            res_val,_= sess.run([val_summary, val_g_loss], feed_dict = feed)
            print('D Cost:', test_d_cost/n_test,'G Cost:',test_g_cost/n_test)
            file_writer.add_summary( res_val, (epoch+1))
        print('Optimization Finished')
        sess.close()

