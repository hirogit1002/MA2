import tensorflow as tf
import tensorboard as tb 
import numpy as np
from layerfunctions import*



def AE(x, keep_prob, batch_size, latent_size, Training, lr, reuse=False):
    with tf.variable_scope("AE") as scope:
        if (reuse):
            scope.reuse_variables()
        flat = encoder(x,Training)
        flat = tf.layers.flatten(flat)
        z = fullyConnected(flat, name='z', output_size=latent_size)
        output = decoder(z,Training, batch_size)
        cross_entropy = -1. * x * tf.log(output + 1e-10) - (1. - x) * tf.log(1. - output + 1e-10)
        loss = tf.reduce_sum(cross_entropy)
        if(reuse):
            return output, loss, z
        else:
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            return output, loss, optimizer, z

def VAE(x, keep_prob, batch_size, latent_size, Training,lr, reuse=False):
    with tf.variable_scope("VAE") as scope:
        if (reuse):
            scope.reuse_variables()
        flat = encoder(x,Training)
        flat = tf.layers.flatten(flat)
        z_mean = fullyConnected(flat, name='z_mean', output_size=latent_size, activation = 'linear')
        z_log_sigma_sq = fullyConnected(flat, name='z_log_sigma_sq', output_size=latent_size, activation = 'linear')
        z = variational(z_mean, z_log_sigma_sq, batch_size, latent_size)
        output = decoder(z,Training, batch_size)
        if(reuse):
            loss = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean, lr, reuse)
            return output, loss, z
        else:
            loss, optimizer = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean, lr, reuse)
            return output, loss, optimizer, z
    
def VAE_test(x, keep_prob, batch_size, latent_size, Training,lr,reuse=False):
    with tf.variable_scope("VAE", reuse=tf.AUTO_REUSE):
        flat = encoder(x,Training)
        flat = tf.layers.flatten(flat)
        z_mean = fullyConnected(flat, name='z_mean', output_size=latent_size, activation = 'linear')
        z_log_sigma_sq = fullyConnected(flat, name='z_log_sigma_sq', output_size=latent_size, activation = 'linear')
        output = decoder(z_mean,Training, batch_size)
        loss, optimizer = create_loss_and_optimizer(x, output, z_log_sigma_sq, z_mean,lr,reuse)
        loss_ext = loss
        return output, loss, optimizer, z_mean    

def DCGAN(x,z,Training, Batch_size, lr,reuse=False):
    generated = generator(z, Training, Batch_size,reuse)
    sig, D_logits, _ = discriminator(x,Training, reuse)
    sig_, D_logits_, _ = discriminator(generated, Training, reuse=True)
    loss_real, loss_fake, g_loss = loss_gan(D_logits,D_logits_)                    
    d_loss = loss_real + loss_fake
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    
    # Optimizer
    if(reuse):
        return d_loss, g_loss, sig, sig_
    else:
        dis_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_vars)
        gen_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_vars)
        return generated, gen_op, dis_op, d_loss, g_loss
    
    
def encoder(x,Training, Name=''):
    p1 = conv2d(x, (Name+'conv1'), [5, 5, 1, 64], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    p2 = conv2d_nrm(p1, (Name+'conv2'), [5, 5, 64, 128], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    p3 = conv2d_nrm(p2, (Name+'conv3'), [5, 5, 128, 256], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    p4 = conv2d_nrm(p3, (Name+'conv4'), [3, 3, 256, 512], Training, [1, 2, 2, 1], 'SAME' ,activation = 'lrelu')
    #flat = tf.layers.flatten(p4)
    return p4#flat

def decoder(z,Training, batchsize, Name='',actf_output='sigmoid'):
    fc1 = fullyConnected(z, name=(Name+'fc1'), output_size=4*4*1024)
    r1 = tf.reshape(fc1, shape=[-1,4,4,1024])
    dc0 = deconv2d_nrm(r1, (Name+'deconv0'), [3, 3, 512, 1024], Training, batchsize, [1, 2, 2, 1], 'SAME', 'relu')
    dc1 = deconv2d_nrm(dc0, (Name+'deconv1'), [3, 3, 256, 512], Training, batchsize, [1, 2, 2, 1], 'SAME', 'relu')
    dc2 = deconv2d_nrm(dc1, (Name+'deconv2'), [5, 5, 128, 256],Training, batchsize, [1, 2, 2, 1], 'SAME', 'relu')
    output = deconv2d(dc2, (Name+'deconv3'), [5, 5, 1, 128],Training, batchsize, [1, 2, 2, 1], 'SAME', actf_output)
    return output


def generator(z, Training, batchsize, reuse=False):
    with tf.variable_scope("generator") as scope:
        if (reuse):
            scope.reuse_variables()
        output = decoder(z,Training, batchsize, 'g_','tanh')
        return output


def discriminator(x, Training, reuse=False, pool_typ = 'max'):
    pooling = {'max':maxpool2d,'avg':avgpool2d}
    with tf.variable_scope("discriminator") as scope:
        if (reuse):
            scope.reuse_variables()
        encoded = encoder(x,Training,'d_')
        extracted = pooling[pool_typ](encoded)
        vectors = tf.layers.flatten(extracted)
        flat = tf.layers.flatten(encoded)
        fc_class = fullyConnected(flat, name='d_fc_class', output_size=1, activation = 'linear')
        sig = tf.nn.sigmoid(fc_class)
        return sig, fc_class, vectors
    
def variational(z_mean, z_log_sigma_sq, batch_size, latent_size):
    eps = tf.random_normal([batch_size, latent_size], 0.0, 1.0, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z 

def create_loss_and_optimizer(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean,lr,reuse):
    inputs = tf.layers.flatten(inputs) 
    x_reconstr_mean = tf.layers.flatten(x_reconstr_mean) 
    loss = vae_loss_cal(inputs, x_reconstr_mean, z_log_sigma_sq, z_mean, epsilon=1e-10)
    if(reuse):
        return loss
    else:
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
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

