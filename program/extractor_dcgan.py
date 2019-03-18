from PIL import Image, ImageFilter
import numpy as np
from PIL import ImageFile
from model import*
from imgproc import*
import glob
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extractor(latent_size,Reuse=False):
    weight_path_ext = '../weigths/'+'DCGAN'+'_'+str(latent_size) + '.ckpt'
    imgs_path = np.array(sorted(glob.glob('../data_test/*.jpg')))
    imgs = np.array([np.array(Image.open(i).convert('L')) for i in imgs_path])
    imgs = norm_intg(imgs,'tanh')
 
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='InputData')
    Training = tf.placeholder(dtype=tf.bool, name='Training')
    _,_, encoder = discriminator(x, Training, reuse=Reuse)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess_ext:
        sess_ext.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess_ext, weight_path_ext)
        vectors = []
        for i in imgs:
            img = i[np.newaxis,:]
            vector = encoder.eval(feed_dict={x: img, Training:False})
            vectors +=[vector]
        vectors = np.array(vectors)
        sess_ext.close()
    vectors= vectors[:,0,:]
    save_path='../save/dcgan_vector'+'_'+str(latent_size) + '.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(vectors, f)





