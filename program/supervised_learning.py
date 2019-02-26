from PIL import Image, ImageFilter
import numpy as np
from PIL import ImageFile
from sklearn.svm import SVC
from sklearn import datasets, svm, metrics
from skimage.transform import resize
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

def split(n, vector_intg, y, test_size):
    n_test = int(n*0.3)
    return vector_intg[:-n_test], vector_intg[-n_test:], y[:-n_test], y[-n_test:], n_test

class SVM():
    def __init__(self,Kernel='linear',Test_size=0.3):
        print('Data loaded')
        self.vectors, self.y = self.load('ck_vector/feature_values_ck.npy','ck_vector/labels.npy')
        self.n = len(self.vectors)
        self.perm = np.random.permutation(self.n)
        self.X_train, self.X_test, self.y_train, self.y_test, self.n_test = split(self.vectors[self.perm], self.y[self.perm], Test_size)
        print('Construct SVCs')
        self.model = SVC(kernel=Kernel, random_state=None,gamma='auto')
        print('Finish construction SVCs')
        
    def fit(self):
        self.model.fit(X=self.X_train, y=self.y_train)
    
    def predict(self):
        print(self.model.score(X=self.X_test, y=self.y_test))
        return self.model.predict(self.X_test)
        
    def load(self,path_vector,path_labels):
        vectors = np.load(path_vector)
        y = np.load(path_labels)
        return vectors, y


class SVM_aff():
    def __init__(self,Kernel='linear',Test_size=0.3):
        self.emos=['neutral','happy','sad','surprise','fear','disgust','anger']
        self.output = [self.load(emo) for emo in self.emos]
        self.d = len(self.output[0][0])
        self.imgpaths = []
        self.img7 =[]
        self.vector7=[]
        for (k, i) in enumerate(self.output):
            self.imgpaths += [np.array([j for j in i[0]])] 
            self.img7 += [np.array([resize(np.array(Image.open(j)), (64, 64),mode= 'reflect') for j in i[0]])] 
            self.vector7 += [np.array([j for j in i[1]])]
        self.vector_intg = []
        self.img_intg = []
        self.y = []
        self.emo_l = []
        for i in self.vector7:
            self.vector_intg.extend(i)
        self.vector_intg = np.array(self.vector_intg)
        for i in self.img7:
            self.img_intg.extend(i)
        #self.y = np.array([np.ones(self.d)*float(i) for (i, x) in enumerate(self.emos)]).flatten()
        for (k, i) in enumerate(self.vector7):
            l = len(i)
            self.emo_l += [l]
            self.y.extend(np.ones(l)*float(k))
        self.y = np.array(self.y)   
        self.n = len(self.vector_intg)
        self.perm = np.random.permutation(self.n)
        self.X_train, self.X_test, self.y_train, self.y_test, self.n_test = split(self.n, self.vector_intg[self.perm], self.y[self.perm], test_size=Test_size)
        self.n_test = len(self.y_test)
        self.test_imgs = np.array(self.img_intg)[self.perm[-self.n_test:]]
        print('Data loaded')
        print('Construct SVCs')
        self.model = SVC(kernel=Kernel, random_state=None,gamma='auto')
        print('Finish construction SVCs')

    def plot(self,idx,size=(50,20)):
        if (idx=='test'):
            d = len(self.test_imgs)
        else:
            d = self.emo_l[idx]
        h = -(-d //10)
        fig = plt.figure(figsize=size,frameon=False)
        for i in range(d):
            plt.subplot(h, 10, (i+1))
            plt.axis('off')
            if (idx=='test'):
                plt.title((str(i+1)+'Label: '+str(self.y_test[i])))
                plt.imshow(self.test_imgs[i])
            else:
                plt.title(str(i+1))
                plt.imshow(self.img7[idx][i])
        plt.show()
        
    def fit(self):
        self.model.fit(X=self.X_train, y=self.y_train)
    
    def predict(self):
        print(self.model.score(X=self.X_test, y=self.y_test))
        return self.model.predict(self.X_test)
        
    def load(self,emo):
        folder = emo
        path = emo+'/feature_values_'+emo+'.npy'
        vectors = np.load(path)
        path = emo+'/image_path_'+emo+'.npy'
        image_path = np.load(path)
        return image_path, vectors
    