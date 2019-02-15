import tensorflow as tf
import tensorboard as tb 
import numpy as np
from layerfunctions import*



def AE(x, keep_prob, batch_size, latent_size, Training,lr):
    with tf.variable_scope("AE", reuse=tf.AUTO_REUSE):
        flat = encoder(x,Training)
        z = fullyConnected(flat, name='z', output_size=latent_size)
        output = decoder(z,Training)
        cross_entropy = -1. * x * tf.log(output + 1e-10) - (1. - x) * tf.log(1. - output + 1e-10)
        loss = tf.reduce_sum(cross_entropy)
        loss_ext = loss
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        return output, loss, loss_ext, optimizer, z

def VAE(x, keep_prob, batch_size, latent_size, Training,lr):
    with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
        flat = encoder(x,Training)
        z_mean = fullyConnected(flat, name='z_mean', output_size=latent_size, activation = 'linear')
        z_log_sigma_sq = fullyConnected(flat, name='z_log_sigma_sq', output_size=latent_size, activation = 'linear')
        z = variational(z_mean, z_log_sigma_sq, batch_size, latent_size)
        output = decoder(z,Training)
        loss, optimizer = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean, lr)
        loss_ext = loss
        return output, loss, loss_ext, optimizer, z    
    
def VAE_test(x, keep_prob, batch_size, latent_size, Training,lr):
    with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
        flat = encoder(x,Training)
        z_mean = fullyConnected(flat, name='z_mean', output_size=latent_size, activation = 'linear')
        z_log_sigma_sq = fullyConnected(flat, name='z_log_sigma_sq', output_size=latent_size, activation = 'linear')
        output = decoder(z_mean,Training)
        loss, optimizer = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean,lr)
        loss_ext = loss
        return output, loss, loss_ext, optimizer, z_mean    

def DCGAN():
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    z = tf.placeholder(tf.float32, [None, latent_size], name='latent')
    Training = tf.placeholder(dtype=tf.bool, name='LabelData')
    generated = generator(z, Training)
    sig, D_logits = discriminator(x,Training, reuse=False)
    sig_, D_logits_ = discriminator(generated, Training, reuse=True)
    loss_real, loss_fake, g_loss = loss_gan(D_logits,D_logits_)                    
    d_loss = loss_real + loss_fake

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    
    # Optimizer
    dis_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_vars)
    gen_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_vars)
    return generated, gen_op, dis_op, d_loss, g_loss, d_loss, g_loss
    
    
def encoder(x,Training, Name=''):
    p1 = conv2d_norm(x, (Name+'conv1'), [5, 5, 1, 64], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    p2 = conv2d_norm(p1, (Name+'conv2'), [5, 5, 64, 128], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    p3 = conv2d_norm(p2, (Name+'conv3'), [5, 5, 128, 256], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    p4 = conv2d_norm(p3, (Name+'conv4'), [3, 3, 256, 512], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    flat = tf.layers.flatten(p4)
    return flat

def decoder(z,Training, Name='',actf_output='sigmoid'):
    fc1 = fullyConnected(z, name=(Name+'fc1'), output_size=4*4*1024)
    r1 = tf.reshape(fc1, shape=[-1,4,4,1024])
    dc1 = deconv2d_norm(r1, (Name+'deconv1'), [3,3], 512,Training, [2, 2], 'relu', 'SAME')
    dc2 = deconv2d_norm(dc1, (Name+'deconv2'), [5,5], 256,Training, [2, 2], 'relu', 'SAME')
    dc3 = deconv2d_norm(dc2, (Name+'deconv3'), [5,5], 128,Training, [2, 2], 'relu', 'SAME')
    output = deconv2d_norm(dc3, (Name+'deconv4'), [5,5], 1,Training, [2, 2], actf_output, 'SAME')
    return output


def generator(z, Training):
    with tf.variable_scope("generator") as scope:
        output = decoder(z,Training, 'g_','tanh')
        return output


def discriminator(x, Training, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        flat = encoder(x,Training,'d_')
        fc_class = fullyConnected(flat, name='d_fc_class', output_size=1, activation = 'linear')
        sig = tf.nn.sigmoid(fc_class)
        return sig, fc_class
    
def variational(z_mean, z_log_sigma_sq, batch_size, latent_size):
    eps = tf.random_normal([batch_size, latent_size], 0.0, 1.0, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z 

def create_loss_and_optimizer(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean,lr):
    inputs = tf.layers.flatten(inputs) 
    x_reconstr_mean = tf.layers.flatten(x_reconstr_mean) 
    loss = vae_loss_cal(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    return loss, optimizer

def vae_loss_cal(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean, epsilon=1e-10):
    reconstr_loss = -tf.reduce_sum(inputs * tf.log(tf.clip_by_value(x_reconstr_mean, 1e-10, 1.0)) + (1.0 - inputs) * tf.log(tf.clip_by_value(1.0 - x_reconstr_mean, 1e-10, 1.0)),1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
    return loss

def loss_gan(D_logits,D_logits_):
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_logits_)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_logits_)))
    return loss_real, loss_fake, g_loss

