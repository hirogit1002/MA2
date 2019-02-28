from PIL import Image, ImageFilter
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import numpy as np
from PIL import ImageFile
from sklearn.svm import SVC
from skimage.transform import resize
from sklearn.manifold import TSNE
from layerfunctions import*
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

emos_inv = {1:'anger',2:'contempt',3:'disgust',4:'fear',5:'happy',6:'sad',7:'surprise'}
emos_idx = [1,2,3,4,5,6,7]

def split(n, vector_intg, y, y_value, img ,test_size):
    n_test = int(n*test_size)
    return vector_intg[:-n_test], vector_intg[-n_test:], y[:-n_test], y[-n_test:], y_value[:-n_test], y_value[-n_test:],img[:-n_test], img[-n_test:], n_test

def CV(model,num):
    maximum = 0.
    minimum = 100.
    avg = 0.
    scores =[]
    for i in range(num):
        model.cv_again()
        model.fit()
        _, score = model.predict()
        maximum = max(maximum,score)
        minimum = min(minimum,score)
        avg += score
        scores +=[score]
    print('Average: ', avg/num)
    print('Maximum: ', maximum)
    print('Minimum: ',minimum)
    return scores

def cv(vectors, y, y_value ,imgs, Test_size=0.3):
    idx = [np.where(y==i)[0] for i in emos_idx]
    X_Train, X_Test, y_Train, y_Test, y_value_Train, y_value_Test, img_Train, img_Test, n_Test, Perm = [], [], [], [], [], [], [], [], [], []
    for i in idx:
        n = len(i)
        vectors_emo = vectors[i]
        y_emo = y[i]
        y_value_emo = y_value[i]
        imgs_emo = imgs[i]
        perm = np.random.permutation(n)
        X_train, X_test, y_train, y_test, y_value_train, y_value_test, img_train, img_test, n_test=split(n ,vectors_emo[perm], y_emo[perm], y_value_emo[perm] ,imgs_emo[perm], Test_size)
        X_Train.extend(X_train)
        X_Test.extend(X_test)
        y_Train.extend(y_train)
        y_Test.extend(y_test)
        y_value_Train.extend(y_value_train)
        y_value_Test.extend(y_value_test)
        img_Train.extend(img_train)
        img_Test.extend(img_test)
        n_Test.extend(np.array([n_test]))
        Perm.extend(perm)
    return np.array(X_Train), np.array(X_Test), np.array(y_Train), np.array(y_Test), np.array(y_value_Train), np.array(y_value_Test), np.array(img_Train), np.array(img_Test), np.array(n_Test), np.array(Perm)


def load(path_vector,path_y_value,path_labels):
    with open(path_vector, 'rb') as f:
        vectors = pickle.load(f)
    with open(path_y_value, 'rb') as f:
        y_values = pickle.load(f)
    path_label = np.array(sorted(glob.glob((path_labels))))
    y = np.array([np.array(pd.read_csv(i,header=None)[0]) for i in path_label])
    return vectors,y_values ,y


class Finetuning():
    def __init__(self,path_vector,path_y_value,path_labels,Kernel='linear',Test_size=0.3):
        self.emos_inv = {1:'anger',2:'contempt',3:'disgust',4:'fear',5:'happy',6:'sad',7:'surprise'}
        self.imgs_path = np.array(sorted(glob.glob('../data_test/*.jpg')))
        self.imgs = np.array([np.array(Image.open(i).convert('L')) for i in self.imgs_path])
        self.vectors, self.y_value, self.y = self.load(path_vector,path_y_value,path_labels)
        with tf.variable_scope("Finetuning") as scope:
            if (reuse):
                scope.reuse_variables()
            fc1 = fullyConnected(flat, name='fc1', output_size=50)
            fc2 = fullyConnected(fc1, name='fc2', output_size=classnum)

class SVM():
    def __init__(self,path_vector,path_y_value,path_labels,Kernel='linear',Test_size=0.3):
        self.emos_inv = {1:'anger',2:'contempt',3:'disgust',4:'fear',5:'happy',6:'sad',7:'surprise'}
        print('Data loaded')
        self.imgs_path = np.array(sorted(glob.glob('../data_test/*.jpg')))
        self.imgs = np.array([np.array(Image.open(i).convert('L')) for i in self.imgs_path])
        self.vectors, self.y_value, self.y = load(path_vector,path_y_value,path_labels)
        self.y_value = np.array(self.y_value)
        self.y = np.array(self.y[:,0])
        self.n = len(self.vectors)
        #self.perm = np.random.permutation(self.n)
        #self.X_train, self.X_test, self.y_train, self.y_test, self.y_value_train, self.y_value_test, self.img_train, self.img_test, self.n_test = split(self.n ,self.vectors[self.perm], self.y[self.perm], self.y_value[self.perm] ,self.imgs[self.perm], Test_size)
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_value_train, self.y_value_test, self.img_train, self.img_test, self.n_test, self.perm = cv(self.vectors, self.y, self.y_value ,self.imgs, Test_size)
        print('Construct SVCs')
        self.model = SVC(kernel=Kernel, random_state=None,gamma='auto')
        print('Finish construction SVCs')
        
    def fit(self):
        self.model.fit(X=self.X_train, y=self.y_train)
    
    def predict(self):
        score = self.model.score(X=self.X_test, y=self.y_test)
        return self.model.predict(self.X_test), score

    def visualize(self, pca=True, size=(20,20)):
        if(pca):
            decomp = PCA(n_components=30).fit_transform(self.vectors)
            z_tsne = TSNE(n_components=2, random_state=0).fit_transform(decomp)
        else:
            z_tsne = TSNE(n_components=2, random_state=0).fit_transform(self.vectors)

        anger = z_tsne[np.where(self.y==1.)[0]]
        contempt = z_tsne[np.where(self.y==2.)[0]]
        disgust = z_tsne[np.where(self.y==3.)[0]]
        fear = z_tsne[np.where(self.y==4.)[0]]
        happy = z_tsne[np.where(self.y==5.)[0]]
        sad = z_tsne[np.where(self.y==6.)[0]]
        surprise = z_tsne[np.where(self.y==7.)[0]]

        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(1,1,1)
        ax.scatter(anger[:, 0],anger[:, 1], c='red', marker='^', label='anger')
        ax.scatter(contempt[:, 0],contempt[:, 1], c='black',marker='x', label='contempt')
        ax.scatter(disgust[:, 0],disgust[:, 1], c='blue',marker='o', label='disgust')
        ax.scatter(fear[:, 0],fear[:, 1], c='green',marker='s', label='fear')
        ax.scatter(happy[:, 0],happy[:, 1], c='yellow',marker='o', label='happy')
        ax.scatter(sad[:, 0],sad[:, 1], c='blue',marker='s', label='sad')
        ax.scatter(surprise[:, 0], surprise[:, 1], c='red',marker='.', label='surprise')
        ax.set_title('Distribution of emotion')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper left')
        fig.show()       
        
        
    def imshow(self,plot='reconst', size=(50,40)):
        n=self.n_test.sum()
        h =-(-n//10)
        if(plot=='reconst'):
            value = self.y_value_test
        else:
            value = self.img_test
        fig = plt.figure(figsize=size)
        for i in range(n):
            plt.subplot(h, 10, (i+1))
            plt.title((str(i+1)+' Label: '+self.emos_inv[int(self.y_test[i])]))
            plt.imshow(value[i],cmap='gray')
        plt.show()
        
        
    def cv_again(self, Test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_value_train, self.y_value_test, self.img_train, self.img_test, self.n_test, self.perm = cv(self.vectors, self.y, self.y_value ,self.imgs, Test_size)
        
    #def split_again(self, Test_size=0.3):
    #    self.perm = np.random.permutation(self.n)
    #    self.X_train, self.X_test, self.y_train, self.y_test, self.y_value_train, self.y_value_test, self.img_train, self.img_test, self.n_test = split(self.n ,self.vectors[self.perm], self.y[self.perm], self.y_value[self.perm] ,self.imgs[self.perm], Test_size)
        