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

    
def train_network(data, test_size, batch_size,init,latent_size, normalizarion,shp, epochs, model_name, logs_path, device):
    models = {'AE':AE,'VAE':VAE}
    tf.reset_default_graph()
    perm = np.random.permutation(len(data))
    data = np.array(data[perm])
    train_data = np.array(data[:-test_size])
    test_data = np.array(data[-test_size:])
    weight_path = '../weigths/'+model_name + '.ckpt'
    n = len(train_data)
    n_test = len(test_data)
    data_shape = shp
    data_shape[0] = None
    x = tf.placeholder(tf.float32, data_shape, name='InputData')
    keep_prob = tf.placeholder(tf.float32)
    Batch_size = tf.placeholder(tf.int32)
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    if(device=='none'):
        out, cost_trn, cost_val, optimizer, fv = models[model_name](x, keep_prob, Batch_size, latent_size, Training)
    else:
        with tf.device(device):
            out, cost_trn, cost_val, optimizer, fv = models[model_name](x, keep_prob, Batch_size, latent_size, Training)
    with tf.name_scope('training'):
        tf.summary.scalar('loss', cost_trn)
    with tf.name_scope('validation'):
        tf.summary.scalar('loss', cost_val)

    trn_summary = tf.summary.merge_all(scope='training')
    val_summary = tf.summary.merge_all(scope='validation')

    n_epochs = epochs
    with tf.Session() as sess:
        if tf.gfile.Exists(logs_path):
            tf.gfile.DeleteRecursively(logs_path) # ./logdirが存在する場合削除
        file_writer = tf.summary.FileWriter(logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        # create log writer object
        saver = tf.train.Saver()
        if (init==False):
            saver.restore(sess, weight_path)
        for epoch in range(n_epochs):
            avg_cost = 0
            sum_loss = 0
            n_batches = int(n / batch_size)
            # Loop over all batches
            counter = 0
            train_data = np.random.permutation(train_data)
            for i in range(n_batches):
                batch_x = train_data[i*batch_size:(i+1)*batch_size]
                imgs = np.array([np.array(Image.open(i).convert('L')) for i in batch_x])
                #imgs = np.array([cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY) for i in batch_x])
                imgs = imgs[:,:,:,np.newaxis]
                if (normalizarion):
                    imgs = imgs/255.
                # Run optimization op (backprop) and cost op (to get loss value)
                _,res_trn ,train_cost = sess.run([optimizer, trn_summary, cost_trn], feed_dict={x: imgs, keep_prob:0.75,Training:True, Batch_size:batch_size})
                avg_cost += train_cost  / n_batches
                sum_loss += train_cost
                sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(counter+1, np.int(n/batch_size)+1, sum_loss/(i+1)))
                sys.stdout.flush()
                counter +=1
            file_writer.add_summary(res_trn, (epoch+1))
            saver.save(sess, weight_path)

            # Display logs per epoch step
            print('Epoch', epoch+1, ' / ', n_epochs, 'cost:', avg_cost)
            print('')
            test_imgs = np.array([np.array(Image.open(i).convert('L')) for i in test_data])
            test_imgs = test_imgs[:,:,:,np.newaxis]
            if (normalizarion):
                test_imgs = test_imgs/255.
            res_val, test_cost =sess.run([val_summary, cost_val], feed_dict={x: test_imgs, keep_prob:1.,Training:False, Batch_size:n_test})
            print('Cost:', test_cost)
            file_writer.add_summary( res_val, (epoch+1))
        print('Optimization Finished')
        y_value = sess.run([out], feed_dict={x: test_imgs, keep_prob:1.,Training:False, Batch_size:n_test})
        sess.close()
        if (normalizarion):
            y_value = y_value*255
    return y_value,out
