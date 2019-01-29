import tensorflow as tf
import numpy as np
import scipy.ndimage.filters as fi

def conv2d(x, name, kshape, strides=[1, 1, 1, 1], pad='SAME' ,activation = 'relu'):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    W = tf.get_variable(name='w_'+name, shape=kshape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name='b_' + name, shape=[kshape[3]],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    out = tf.nn.conv2d(x,W,strides=strides, padding=pad)
    out = tf.nn.bias_add(out, b)
    out = actdict[activation](out)
    return out
# ---------------------------------
def deconv2d(x, name, kshape, n_outputs, strides=[1, 1],activation = 'relu',pad='SAME'):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    out = tf.contrib.layers.conv2d_transpose(x,
                                             num_outputs= n_outputs,
                                             kernel_size=kshape,
                                             stride=strides,
                                             padding=pad,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                             biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                             activation_fn=actdict[activation])
    return out


def conv2d_norm(x, name, kshape, Training, strides=[1, 1, 1, 1], pad='SAME' ,activation = 'relu'):
    actdict = {'relu':tf.nn.relu,'tanh':tf.nn.tanh, 'linear':tf.identity,'sigmoid':tf.nn.sigmoid,'lrelu':lrelu}
    W = tf.get_variable(name='w_'+name, shape=kshape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name='b_' + name, shape=[kshape[3]],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    out = tf.nn.conv2d(x,W,strides=strides, padding=pad)
    out = tf.nn.bias_add(out, b)
    norm = tf.layers.batch_normalization(out,training=Training)
    outnorm = actdict[activation](norm)
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
    W = tf.get_variable(name='w_'+name,
                        shape=[input_size, output_size],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name='b_'+name,
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

def gkern(ch, kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros.
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    gaus = fi.gaussian_filter(inp, nsig)[:,:,np.newaxis,np.newaxis]
    return tf.cast(tf.Variable((np.ones((kernlen,kernlen,ch,ch),np.float32)*gaus), name='gkern'), tf.float32)