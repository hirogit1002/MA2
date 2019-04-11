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
    weight_path = '../weigths/'+'DCGAN'+'_'+str(latent_size) + '.ckpt'
    n = len(train_data)
    n_test = len(test_data)
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    z = tf.placeholder(tf.float32, [None, latent_size], name='latent')
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    Batch_size = tf.placeholder(tf.int32)
    generated, gen_op, dis_op, d_loss, g_loss = DCGAN(x,z,Training,batch_size, lr)
    val_d_loss, val_g_loss, val_D_logits, val_D_logits_ = DCGAN(x,z,Training,n_test, lr, reuse=True)
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
        try:
            sess.run(tf.global_variables_initializer())
        except:
            sess.run(tf.initialize_all_variables())
        # create log writer object
        saver = tf.train.Saver()
        if (init==False):
            saver.restore(sess, weight_path)            
        start_time = time.time()
        epoch_time =0.
        k = 1.
        f= open("../accuracy_discriminator.txt","w+")
        for epoch in range(epochs):
            epoch_start = time.time()
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
                    imgs = imgs[:,:,:,np.newaxis].astype(np.float32)
                # Run optimization op (backprop) and cost op (to get loss value)
                feed = {z: sample_z(batch_size, latent_size), x: imgs, Training:True}      
                _,train_d_loss = sess.run([dis_op, d_loss], feed_dict=feed )
                _,train_g_loss = sess.run([gen_op, g_loss], feed_dict=feed )
                _,res_trn,_ = sess.run([gen_op, trn_summary,g_loss], feed_dict=feed)
                sum_d_loss += (train_d_loss / n_batches)
                sum_g_loss += (train_g_loss/ n_batches)
                sys.stdout.write("\r%s" % "batch: {}/{}, d_loss: {}, g_loss: {}, time: {}".format(counter+1, np.int(n/batch_size)+1, sum_d_loss/(i+1), sum_g_loss/(i+1),(time.time()-start_time)))
                sys.stdout.flush()
                #print("\r%s" % "batch: {}/{}, d_loss: {}, g_loss: {}, time: {}".format(counter+1, np.int(n/batch_size)+1, sum_d_loss/(i+1), sum_g_loss/(i+1),(time.time()-start_time)))
                counter +=1
            file_writer.add_summary(res_trn, (epoch+1))
            saver.save(sess, weight_path)

            # Display logs per epoch step
            print('')
            print('Epoch', epoch+1, ' / ', epochs, 'D Cost:', sum_d_loss/epochs, 'G Cost:', sum_g_loss/epochs)
            test_imgs = np.array([np.array(Image.open(i).convert('L')) for i in test_data])
            if (normalizarion):
                test_imgs = norm_intg(test_imgs,'tanh')
            else:
                test_imgs =  test_imgs[:,:,:,np.newaxis].astype(np.float32)                
            feed = {z: sample_z(n_test, latent_size), x: test_imgs, Training:False}  
            test_d_cost, test_g_cost,d_real,d_fake = sess.run([val_d_loss,val_g_loss, val_D_logits, val_D_logits_], feed_dict = feed)
            res_val,_= sess.run([val_summary,val_d_loss], feed_dict = feed)
            print('D Cost:', test_d_cost/n_test,'G Cost:',test_g_cost/n_test)
            flat_d_real,flat_d_fake = d_real.flatten(),d_fake.flatten()
            d_length = len(flat_d_real)
            ones = np.ones(d_length,np.int)
            zeros = np.zeros(d_length,np.int)
            accuracy=(((np.append(flat_d_real,flat_d_fake)>=0.5).astype(np.int))==np.append(ones,zeros)).sum()/(d_length+d_length)
            print('Accuracy of the Discriminator:', accuracy)
            f.write("%f\r" % accuracy)
            file_writer.add_summary( res_val, (epoch+1))
            epoch_end = time.time()-epoch_start
            epoch_time+=epoch_end
            print('Time per epoch: ',(epoch_time/k),'s/epoch')
            print('')
            k+=1.
        f.close() 
        print('Optimization Finished with time: ',(time.time()-start_time))
        sess.close()

