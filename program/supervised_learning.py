from PIL import Image, ImageFilter
import numpy as np
from PIL import ImageFile
from sklearn.svm import SVC
from skimage.transform import resize
from sklearn.manifold import TSNE
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

def split(n, vector_intg, y, y_value, img ,test_size):
    n_test = int(n*0.3)
    return vector_intg[:-n_test], vector_intg[-n_test:], y[:-n_test], y[-n_test:], y_value[:-n_test], y_value[-n_test:],img[:-n_test], img[-n_test:], n_test

class SVM():
    def __init__(self,path_vector,path_y_value,path_labels,Kernel='linear',Test_size=0.3):
        self.emos_inv = {1:'anger',2:'contempt',3:'disgust',4:'fear',5:'happy',6:'sad',7:'surprise'}
        print('Data loaded')
        self.imgs_path = np.array(sorted(glob.glob('../data_test/*.jpg')))
        self.imgs = np.array([np.array(Image.open(i).convert('L')) for i in self.imgs_path]
        self.vectors, self.y_value, self.y = self.load(path_vector,path_y_value,path_labels)
        self.y_value = np.array(self.y_value)
        self.y = np.array(self.y[:,0])
        self.n = len(self.vectors)
        self.perm = np.random.permutation(self.n)
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_value_train, self.y_value_test, self.img_train, self.img_test, self.n_test = split(self.n ,self.vectors[self.perm], self.y[self.perm], self.y_value[self.perm] ,self.imgs[self.perm], Test_size)
        print('Construct SVCs')
        self.model = SVC(kernel=Kernel, random_state=None,gamma='auto')
        print('Finish construction SVCs')
        
    def fit(self):
        self.model.fit(X=self.X_train, y=self.y_train)
    
    def predict(self):
        print(self.model.score(X=self.X_test, y=self.y_test))
        return self.model.predict(self.X_test)
        
    def load(self,path_vector,path_y_value,path_labels):
        with open(path_vector, 'rb') as f:
            vectors = pickle.load(f)
        with open(path_y_value, 'rb') as f:
            y_values = pickle.load(f)
        path_label = np.array(sorted(glob.glob((path_labels))))
        y = np.array([np.array(pd.read_csv(i,header=None)[0]) for i in path_label])
        return vectors,y_values ,y

    def visualize(self):
        z_tsne = TSNE(n_components=2, random_state=0).fit_transform(self.vectors)
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1])
        plt.show()
    
    def imshow(self,plot='reconst', size=(50,40)):
        n=self.n_test
        h =-(-n//10)
        if(plot=='reconst'):
            value = y_value_test
        else:
            value = self.img_test
        fig = plt.figure(figsize=size)
        for i in range(n):
            plt.subplot(h, 10, (i+1))
            plt.title((str(i+1)+' Label: '+self.emos_inv[int(self.y_test[i])]))
            plt.imshow(self.y_value_test[i],cmap='gray')
        plt.show()
        
        