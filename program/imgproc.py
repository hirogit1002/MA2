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

def img_patch_old(img, patch_size):
    patches = []
    img= np.array(img)
    row = img.shape[0]
    col = img.shape[1]
    row_ite = row // patch_size
    col_ite = col // patch_size
    for i in range(row_ite):
        for j in range(col_ite):
            patch = img[patch_size*i:patch_size*(i+1),patch_size*j:patch_size*(j+1)]
            patches +=[patch]
    return np.array(patches)

def img_patch(imgs, patch_size):
    imgs= np.array(imgs)
    row = imgs.shape[1]
    col = imgs.shape[2]
    row_ite = row // patch_size
    col_ite = col // patch_size
    patches = np.empty((len(imgs),row_ite*col_ite,patch_size,patch_size))
    k=0
    for i in range(row_ite):
        for j in range(col_ite):
            patches[:,k] = imgs[:,patch_size*i:patch_size*(i+1),patch_size*j:patch_size*(j+1)]
            k+=1
    return patches

def patch_intg(patchs,mul):
    pshp=patchs.shape
    psize = pshp[2]
    img = np.empty((pshp[0],psize*mul,psize*mul))
    k=0
    for i in range(mul):
        for j in range(mul):
            img[:,i*psize :(i+1)*psize ,j*psize :(j+1)*psize ] = patchs[:,k,:,:]
            k+=1
    return img

def flatten(patchs):
    pshp=patchs.shape
    flat = np.reshape(patchs,(pshp[0],pshp[1],pshp[2]*pshp[3]))
    return flat, pshp

def reshape(flat,orig_shp):
    flat = np.reshape(flat,(orig_shp[0],orig_shp[1],orig_shp[2],orig_shp[3]))
    return flat

def normalization(img_flattten, const=10.):
    mean = np.mean(img_flattten,axis=2)[:,:,np.newaxis]
    var = np.var(img_flattten,axis=2)[:,:,np.newaxis]
    norm = (img_flattten-mean)/np.sqrt(var+const)
    return norm, mean, var

#def whitning(X):
#    return np.array([whiten(i) for i in X])


def preprocess(X, patch_size,sig=False):
    mul = X.shape[1] // patch_size
    patchs = img_patch(X, patch_size)
    flat, orig_shp = flatten(patchs)
    normd, mean, var = normalization(flat)
    whintend = whitning(normd)
    if(sig):
        whintend = sigmoid(whintend)
    #img_itg_whintend = patch_intg(reshape(whintend,orig_shp),mul)
    return whintend 
    
def denormalization(norm, mean, var, const=10.):
    shp = norm.shape
    norm_flattten=norm.flatten()
    img = norm*np.sqrt(var+const)+mean
    return np.reshape(img,shp).astype(np.int)

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

def make_group(load_path,save_path):
    info = pd.read_csv(load_path, header = None, sep=" ")
    grupe_idx = np.array(info[0])
    group = []
    for i in range(np.max(grupe_idx)+1):
        group +=[np.where(i==grupe_idx)[0]]
    with open(save_path, "wb") as fp:   #Pickling
        pickle.dump(group, fp)

class dataload:
    def __init__(self, path = '../../LY/', group_path = '../../group/group_LY', ext = '.bmp' ):
        with open(group_path, "rb") as fp:   #Pickling
            GP = pickle.load(fp)
        self.GP = GP
        GP_num=len(self.GP)
        n = np.max(self.GP[(GP_num-1)])+1
        self.sortedlis = [(path+str(i)+ext) for i in range(n)]

    def pickup(self):
        idx_ary=np.empty((len(self.GP),2),np.int)
        counter = 0
        for i in self.GP:
            idx_ary[counter] = np.random.choice(i,2, replace=False)
            counter +=1
        self.ramdom_pickedup = idx_ary
        return idx_ary
    
    def load_idx(self):
        sortedlis = np.array(self.sortedlis)
        idx1 = sortedlis[self.ramdom_pickedup[:,0]]
        idx2 = sortedlis[self.ramdom_pickedup[:,1]]
        return idx1, idx2

