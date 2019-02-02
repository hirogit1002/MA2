import tensorflow as tf
import tensorboard as tb 
import numpy as np
from layerfunctions import*



def AE(x, keep_prob, batch_size, latent_size, Training):
    with tf.variable_scope("AE", reuse=tf.AUTO_REUSE):
        with tf.device('/gpu:2'):
            flat = encoder(x,Training)
            z = fullyConnected(flat, name='z', output_size=latent_size)
            output = decoder(z,Training)
            #loss = tf.reduce_mean(tf.square(tf.subtract(output, x)))
            #cross_entropy = -1. * x * tf.log(output + 1e-10) - (1. - x) * tf.log(1. - output + 1e-10)
            cross_entropy =tf.subtract( tf.multiply(tf.multiply(tf.constant([-1.]), x), tf.log(tf.add(output, tf.constant([1e-10])))), tf.multiply(tf.subtract(tf.constant([1.]), x), tf.log(tf.add(tf.subtract(tf.constant([1.]), output),tf.constant([1e-10])))))
            loss = tf.reduce_sum(cross_entropy)
            loss_ext = loss
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
            return output, loss, loss_ext, optimizer, z

def VAE(x, keep_prob, batch_size, latent_size, Training):
    with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
        flat = encoder(x,Training)
        z_mean = fullyConnected(flat, name='z_mean', output_size=latent_size, activation = 'linear')
        z_log_sigma_sq = fullyConnected(flat, name='z_log_sigma_sq', output_size=latent_size, activation = 'linear')
        z = variational(z_mean, z_log_sigma_sq, batch_size, latent_size)
        output = decoder(z,Training)
        loss, optimizer = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean)
        loss_ext = loss
        return output, loss, loss_ext, optimizer, z    
    
def VAE_test(x, keep_prob, batch_size, latent_size, Training):
    with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
        flat = encoder(x,Training)
        z_mean = fullyConnected(flat, name='z_mean', output_size=latent_size, activation = 'linear')
        z_log_sigma_sq = fullyConnected(flat, name='z_log_sigma_sq', output_size=latent_size, activation = 'linear')
        output = decoder(z_mean,Training)
        loss, optimizer = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean)
        loss_ext = loss
        return output, loss, loss_ext, optimizer, z_mean    
    
def encoder(x,Training):
    p1 = conv2d_norm(x, 'conv1', [5, 5, 1, 64], Training, [1, 1, 1, 1], 'SAME' ,activation = 'lrelu')
    pool1 = maxpool2d(p1,'pool1',kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    p2 = conv2d_norm(pool1, 'conv2', [5, 5, 64, 128], Training, [1, 1, 1, 1], 'SAME' ,activation = 'lrelu')
    pool2 = maxpool2d(p2,'pool2',kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    p3 = conv2d_norm(pool2, 'conv3', [5, 5, 128, 256], Training, [1, 1, 1, 1], 'SAME' ,activation = 'lrelu')
    pool3 = maxpool2d(p3,'pool3',kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    p4 = conv2d_norm(pool3, 'conv4', [3, 3, 256, 512], Training, [1, 1, 1, 1], 'SAME' ,activation = 'lrelu')
    pool4 = maxpool2d(p4,'pool4',kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    flat = tf.layers.flatten(pool4)
    return flat

def decoder(z,Training):
    fc1 = fullyConnected(z, name='fc1', output_size=4*4*1024)
    r1 = tf.reshape(fc1, shape=[-1,4,4,1024])
    dc1 = deconv2d_norm(r1, 'deconv1', [3,3], 512,Training, [2, 2], 'relu', 'SAME')
    dc2 = deconv2d_norm(dc1, 'deconv2', [5,5], 256,Training, [2, 2], 'relu', 'SAME')
    dc3 = deconv2d_norm(dc2, 'deconv3', [5,5], 128,Training, [2, 2], 'relu', 'SAME')
    output = deconv2d_norm(dc3, 'deconv4', [5,5], 1,Training, [2, 2], 'sigmoid', 'SAME')
    return output
    
def variational(z_mean, z_log_sigma_sq, batch_size, latent_size):
    eps = tf.random_normal([batch_size, latent_size], 0.0, 1.0, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z 

def create_loss_and_optimizer(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean):
    inputs = tf.layers.flatten(inputs) 
    x_reconstr_mean = tf.layers.flatten(x_reconstr_mean) 
    loss = vae_loss_cal(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    return loss, optimizer

def vae_loss_cal(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean, epsilon=1e-10):
    reconstr_loss = -tf.reduce_sum(inputs * tf.log(tf.clip_by_value(x_reconstr_mean, 1e-10, 1.0)) + (1.0 - inputs) * tf.log(tf.clip_by_value(1.0 - x_reconstr_mean, 1e-10, 1.0)),1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
    return loss

