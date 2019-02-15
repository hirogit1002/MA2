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

    
def train_network(data, test_size, batch_size,init,latent_size, normalizarion, epochs, model_name, logs_path, lr):
    models = {'AE':AE,'VAE':VAE}
    tf.reset_default_graph()
    perm = np.random.permutation(len(data))
    data = np.array(data[perm])
    train_data = np.array(data[:-test_size])
    test_data = np.array(data[-test_size:])
    weight_path = '../weigths/'+model_name + '.ckpt'
    n = len(train_data)
    n_test = len(test_data)
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    keep_prob = tf.placeholder(tf.float32)
    Batch_size = tf.placeholder(tf.int32)
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    out, cost_trn, optimizer, fv = models[model_name](x, keep_prob, batch_size, latent_size, Training,lr)
    out, cost_val, fv = models[model_name](x, keep_prob, n_test, latent_size, Training,lr,True)
    with tf.name_scope('training'):
        tf.summary.scalar('loss', cost_trn)
    with tf.name_scope('validation'):
        tf.summary.scalar('loss', cost_val)

    trn_summary = tf.summary.merge_all(scope='training')
    val_summary = tf.summary.merge_all(scope='validation')

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: #allow_soft_placement=True, 
        if tf.gfile.Exists(logs_path):
            tf.gfile.DeleteRecursively(logs_path) # ./logdirが存在する場合削除
        file_writer = tf.summary.FileWriter(logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        if (init==False):
            saver.restore(sess, weight_path)
        for epoch in range(epochs):
            sum_loss = 0
            n_batches = int(n / batch_size)
            # Loop over all batches
            counter = 0
            train_data = np.random.permutation(train_data)
            for i in range(n_batches):
                batch_x = train_data[i*batch_size:(i+1)*batch_size]
                imgs = np.array([np.array(Image.open(i).convert('L')) for i in batch_x])
                if (normalizarion):
                    imgs = norm_intg(imgs)
                else:
                    imgs = imgs[:,:,:,np.newaxis].astype(np.float32)
                # Run optimization op (backprop) and cost op (to get loss value)
                _,res_trn ,train_cost = sess.run([optimizer, trn_summary, cost_trn], feed_dict={x: imgs, keep_prob:0.75,Training:True, Batch_size:batch_size})
                sum_loss += (train_cost / n_batches)
                sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(counter+1, np.int(n/batch_size)+1, sum_loss/(i+1)))
                sys.stdout.flush()
                counter +=1
            file_writer.add_summary(res_trn, (epoch+1))
            saver.save(sess, weight_path)

            # Display logs per epoch step
            print('')
            print('Epoch', epoch+1, ' / ', epochs, 'Training Loss:', sum_loss/n_batches)
            print('')
            test_imgs = np.array([np.array(Image.open(i).convert('L')) for i in test_data])
            if (normalizarion):
                test_imgs = norm_intg(test_imgs)
            else:
                test_imgs =  test_imgs[:,:,:,np.newaxis].astype(np.float32)  
            res_val, test_cost =sess.run([val_summary, cost_val], feed_dict={x: test_imgs, keep_prob:1.,Training:False, Batch_size:n_test})
            print('Validation Loss:', test_cost/n_test)
            file_writer.add_summary( res_val, (epoch+1))
        print('Optimization Finished')
        sess.close()


