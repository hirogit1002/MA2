import gzip
import numpy as np
import pandas as pd
import glob
import pickle
#from scipy.cluster.vq import whiten

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sample_z(batch_size, latent_size):
    return np.random.uniform(-1., 1., size=[batch_size, latent_size])


def flatten(patchs):
    pshp=patchs.shape
    flat = np.reshape(patchs,(pshp[0],pshp[1]*pshp[2]))
    return flat, pshp

def reshape(flat,orig_shp):
    orig = np.reshape(flat,(orig_shp[0],orig_shp[1],orig_shp[2],1))
    return orig

def normalization(img_flattten, const=10.):
    mean = np.mean(img_flattten,axis=1)[:,np.newaxis]
    var = np.var(img_flattten,axis=1)[:,np.newaxis]
    norm = (img_flattten-mean)/np.sqrt(var+const)
    return norm, mean, var


def norm_intg(imgs,activate='sigmoid'):
    flats, pshp = flatten(imgs)
    norm, mean, var = normalization(flats, const=10.)
    if (activate=='sigmoid'):
        sig = sigmoid(norm)
    else:
        sig = np.tanh(norm)
    return reshape(sig,pshp)



def meanIU(vecpred,norm_true,pn,nn,i):
    TP = (norm_true * vecpred).sum()
    FN = ((((norm_true - vecpred) + 1) / 2).astype(int)).sum()
    FP = (-(((norm_true - vecpred) - 1) / 2).astype(int)).sum()
    MIU = TP / (TP + FP + FN)
    print(i)
    print("FNR: ", FN / pn)
    print("FPR: ", FP / nn)
    print("TPR: ", TP / pn)
    print("mean_IU: ", MIU)
    print(" ")
    return MIU


