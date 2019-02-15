import tensorflow as tf
import numpy as np


def conv2d(x, name, kshape, Training, Strides=[1, 2, 2, 1], pad='SAME',activation = 'relu', stddev=0.02):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    with tf.variable_scope(name):
        w = tf.get_variable('w', kshape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=Strides, padding='SAME')
        biases = tf.get_variable('biases', [kshape[3]], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        outnorm = actdict[activation](conv)
        return outnorm

def conv2d_nrm(x, name, kshape, Training, Strides=[1, 2, 2, 1], pad='SAME',activation = 'relu', stddev=0.02):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    with tf.variable_scope(name):
        w = tf.get_variable('w', kshape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=Strides, padding='SAME')
        biases = tf.get_variable('biases', [kshape[3]], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=Training, scope="batch_norm")
        outnorm = actdict[activation](bn)
        return outnorm
    

def deconv2d(x, name, kshape, Training, Strides=[1, 2, 2, 1], pad='SAME',activation = 'relu', stddev=0.02):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', kshape, initializer=tf.random_normal_initializer(stddev=stddev))
        shp = x.get_shape()
        print(shp)
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=[shp[1]*2,shp[2]*2,kshape[2]], strides=Strides, padding=pad)
        biases = tf.get_variable('biases', [kshape[2]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        outnorm = actdict[activation](deconv)
        return outnorm

def deconv2d_nrm(x, name, kshape, Training, Strides=[1, 2, 2, 1], pad='SAME',activation = 'relu', stddev=0.02):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', kshape, initializer=tf.random_normal_initializer(stddev=stddev))
        shp = x.get_shape()
        print(shp)
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=[np.array(shp[0]),np.array(shp[1]*2),np.array(shp[2]*2),kshape[2]], strides=Strides, padding=pad)
        biases = tf.get_variable('biases', [kshape[2]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        bn = tf.contrib.layers.batch_norm(deconv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=Training, scope="batch_norm")
        outnorm = actdict[activation](bn)
        return outnorm



def conv2d_norm(x, name, kshape, Training, strides=[1, 1, 1, 1], pad='SAME' ,activation = 'relu'):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    W = tf.get_variable(name='w_'+name, shape=kshape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name='b_' + name, shape=[kshape[3]],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    out = tf.nn.conv2d(x,W,strides=strides, padding=pad)
    out = tf.nn.bias_add(out, b)
    mean, variance = tf.nn.moments(out, [0, 1, 2])
    bn = tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5)
    outnorm = actdict[activation](bn)
    return outnorm
# ---------------------------------
def deconv2d_norm(x, name, kshape, n_outputs,Training, strides=[1, 1],activation = 'relu',pad='SAME'):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    out = tf.contrib.layers.conv2d_transpose(x,
                                             num_outputs= n_outputs,
                                             kernel_size=kshape,
                                             stride=strides,
                                             padding=pad,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                             biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                             activation_fn=actdict['linear'])
    
    norm = tf.layers.batch_normalization(out,training=Training)
    outnorm = actdict[activation](norm)  
    return outnorm

#   ---------------------------------
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    out = tf.nn.max_pool(x,ksize=kshape, strides=strides,padding='SAME')
    return out

#   ---------------------------------
def avgpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    out = tf.nn.avg_pool(x,ksize=kshape, strides=strides,padding='SAME')
    return out
#   ---------------------------------
def upsample(x, name, factor=[2,2]):
    size = [int(x.shape[1] * factor[0]), int(x.shape[2] * factor[1])]
    out = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)
    return out
#   ---------------------------------
def fullyConnected(x, name, output_size, activation = 'relu'):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    input_size = x.shape[1:]
    input_size = int(np.prod(input_size))
    W = tf.get_variable(name=name+'_w',
                        shape=[input_size, output_size],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name=name+'_b',
                        shape=[output_size],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    x = tf.reshape(x, [-1, input_size])
    out = actdict[activation](tf.add(tf.matmul(x, W), b))
    return out
#   ---------------------------------
def dropout(x, name, keep_rate):
    out = tf.nn.dropout(x, keep_rate)
    return out


def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, tf.constant(alpha)))

def substructnormal(x, Name, ch, ksize=5, div=1):
    kernel = gkern(ch, kernlen=ksize, nsig=div)
    W = tf.get_variable(name=Name, initializer=kernel)
    mean = tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')
    normed = tf.subtract(x, mean)
    return normed

