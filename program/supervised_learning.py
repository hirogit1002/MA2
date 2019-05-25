from PIL import Image, ImageFilter
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import numpy as np
from PIL import ImageFile
from sklearn.svm import SVC
from skimage.transform import resize
from sklearn.manifold import TSNE
from layerfunctions import*
from model import*
from imgproc import*
import glob
import pickle
import pandas as pd
import seaborn as sns
import time
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from itertools import cycle

emos_inv = {0:'anger',1:'surprise',2:'disgust',3:'fear',4:'happy',5:'sad'}
emos_dict = {'anger':0,'surprise':1,'disgust':2,'fear':3,'happy':4,'sad':5}
emos_idx = [0,1,2,3,4,5]
emos = ['anger','surprise','disgust','fear','happy','sad']
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','black'])

class load():
    def __init__(self, path_vector, path_y_value, path_imgs_train='' ,path_vector_train='', path_y_value_train='',emo_num=6, train_sample_num=100):
        self.vectors = self.load_vectors(path_vector)
        self.y_value = self.load_pred_imgs(path_y_value)
        self.y = self.load_label_test()
        self.y_value = np.array(self.y_value)
        self.imgs = self.load_imgs()
        self.idx_without_contempt = np.where(self.y!=2)[0]
        self.vectors = self.vectors[self.idx_without_contempt]
        self.y = self.y[self.idx_without_contempt]
        self.y_value = self.y_value[self.idx_without_contempt]
        self.imgs = self.imgs[self.idx_without_contempt]
        self.y[np.where(self.y==7)[0]] = 2
        self.y = np.array(self.y[:,0])-1.
        self.n = len(self.vectors)
        if(path_vector_train!=''):
            self.vectors_train = self.load_vectors(path_vector_train)
            self.y_value_train = self.load_pred_imgs(path_y_value_train)
            self.y_train = self.label_gen_train(emo_num,train_sample_num)
            with open(path_imgs_train, 'rb') as f:
                self.imgs_train = pickle.load(f)
            self.X_train = self.vectors_train
            self.X_test = self.vectors
            self.y_test = self.y
            self.y_value_test = self.y_value
            self.img_test = self.imgs
            self.n_test = len(self.imgs)
        
    def output(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.y_value_train, self.y_value_test, self.imgs_train, self.img_test, self.n_test
            
    def show(self,num,images, size=(5,5)):
        fig = plt.figure(figsize=size)
        plt.axis('off')
        plt.imshow(images[num],cmap='gray')
        plt.show()
    def load_imgs(self):
        imgs_path = np.array(sorted(glob.glob('../data_test/*.jpg')))
        imgs = np.array([np.array(Image.open(i).convert('L')) for i in imgs_path]) 
        return imgs

    def load_pred_imgs(self,path_y_value):
        with open(path_y_value, 'rb') as f:
            imgs = pickle.load(f)
        return imgs

    def load_vectors(self,path_vector):
        with open(path_vector, 'rb') as f:
            vectors = pickle.load(f)
        return vectors

    def load_label_test(self,path_labels='../labels/*.txt'):
        path_label = np.array(sorted(glob.glob((path_labels))))
        y = np.array([np.array(pd.read_csv(i,header=None)[0]) for i in path_label])
        return y

    def label_gen_train(self,emo_num,num):
        labels = np.array([])
        for i in range(emo_num):
            labels = np.append(labels,np.ones(num)*i)
        return labels

    
    
def cv_kfold(y, k,emo_num):
    def append(a,k,emo_num):
        leers = []
        for i in range(k):
            leer = np.array([])
            for j in range(emo_num):
                leer =np.append(leer,a[j*k+i])
            leers +=[leer]  
        return leers
    kfd_trn_idx, kfd_tst_idx = [], []
    idx = [np.where(y==i)[0] for i in emos_idx]
    for i in idx:
        n = len(i)
        perm = np.random.permutation(n)
        kf = KFold(n_splits=k)
        kf.get_n_splits(perm)
        shfld = i[perm]
        for train_idx, test_idx in kf.split(shfld):
            kfd_trn_idx += [shfld [train_idx]]
            kfd_tst_idx += [shfld [test_idx]]
    return append(kfd_trn_idx,k,emo_num), append(kfd_tst_idx,k,emo_num)
    
def CV_kfold(data,model,k=10, emo_num=6):
    kfd_trn_idx, kfd_tst_idx = cv_kfold(data.y, k,emo_num)
    maximum = 0.
    minimum = 100.
    avg = 0.
    scores =[]
    perm = []
    AmAP = []
    precisions, recalls =[],[]
    AAPs = np.zeros(emo_num)
    vectors = np.array(data.vectors)
    y = np.array(data.y)
    y_value = np.array(data.y_value)
    imgs = np.array(data.imgs)
    for i in range(k):
        trn_idx = np.array(kfd_trn_idx[i],np.int)
        tst_idx = np.array(kfd_tst_idx[i],np.int)
        X_train = vectors[trn_idx]
        X_test = vectors[tst_idx]
        y_train = y[trn_idx]
        y_test = y[tst_idx]
        y_value_train = y_value[trn_idx]
        y_value_test = y_value[tst_idx]
        img_train = imgs[trn_idx]
        img_test = imgs[tst_idx]
        model.fit(X_train,y_train)
        _, score = model.predict(X_test,y_test)
        mAP, APs, precision, recall = model.evaluate(plot=False)
        idx = np.argmax([score,maximum])
        if(idx):
            perm = (trn_idx,tst_idx)
        maximum = max(maximum,score)
        minimum = min(minimum,score)
        avg += score
        scores +=[score]
        AmAP += [mAP]
        AAPs += np.array(APs)
        precisions +=[precision]
        recalls +=[recall]
    AAPs=AAPs/k
    print('Accuracy')
    print('Average: ', avg/k)
    print('Maximum: ', maximum)
    print('Minimum: ',minimum)
    print('Standard Diviation: ',np.std(scores))
    print('Average mAP: ',np.mean(AmAP))
    print('Standard Diviation(mAP): ',np.std(AmAP))
    print('Emotions',emos)
    print('Average AP: ',AAPs)
    return scores, perm, precisions, recalls


class SVM():
    def __init__(self,Kernel='linear',C=1.0):
        print('Construct SVCs')
        self.model = SVC(kernel=Kernel,C=C ,random_state=None,gamma='auto')
    def evaluate(self,plot=True):
        precision = dict()
        recall = dict()
        average_precision = dict()
        self.bis = []
        for i in set(self.y_test):
            bi=np.zeros(len(self.y_test),np.int)
            bi[np.where(self.y_test==i)[0]] =1
            self.bis+=[bi]
        for i in range(len(set(self.y_test))):
            precision[i], recall[i], _ = precision_recall_curve(self.bis[i],self.value[:, i])
            average_precision[i] = average_precision_score(self.bis[i], self.value[:, i])  
        if(plot):
            plt.figure(figsize=(7, 8))
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            labels = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
            lines.append(l)
            labels.append('iso-f1 curves')
        mAP = 0.
        APs = []
        for i, color in zip(range(len(set(self.y_test))), colors):
            if(plot):
                l, = plt.plot(recall[i], precision[i], color=color, lw=2)
                lines.append(l)
                labels.append('Precision-recall for class: {0} (AP = {1:0.2f})'''.format(emos_inv[i], average_precision[i]))
            APs+=[average_precision[i]]
            mAP+= average_precision[i]
        mAP = mAP/6.
        APs = np.array(APs)
        if(plot):
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.25)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            ttl = 'Precision-Recall curve with mAP = ' + str(mAP)
            plt.title(ttl)
            plt.legend(lines, labels, loc=(1.1, 0.5), prop=dict(size=14))
            plt.show()
        return mAP, APs, precision, recall

    def cmat(self):
        true = [emos_inv[i] for i in self.y_test.astype(np.int)]
        pred = [emos_inv[i] for i in self.pred.astype(np.int)]
        #labels = emos
        labels = sorted(list(set(true)))
        cmx_data = confusion_matrix(true, pred, labels=labels)
        cmx_data =cmx_data/cmx_data.sum(1)[:,np.newaxis]
        df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cmx, annot=True)
        plt.show()
        
    def fit(self,X_train,y_train):
        self.model.fit(X=X_train, y=y_train)
    
    def predict(self,X_test, y_test):
        self.y_test = y_test
        score = self.model.score(X=X_test, y=y_test)
        self.value = self.model.decision_function(X_test)
        self.pred = self.model.predict(X_test)
        return self.pred, score

def visualize(vectors, y,pca=30, size=(20,20)):
    if(pca>0):
        decomp = PCA(n_components=pca).fit_transform(vectors)
        z_tsne = TSNE(n_components=2, random_state=0).fit_transform(decomp)
    else:
        z_tsne = TSNE(n_components=2, random_state=0).fit_transform(vectors)

    anger = z_tsne[np.where(y==0.)[0]]
    disgust = z_tsne[np.where(y==2.)[0]]
    fear = z_tsne[np.where(y==3.)[0]]
    happy = z_tsne[np.where(y==4.)[0]]
    sad = z_tsne[np.where(y==5.)[0]]
    surprise = z_tsne[np.where(y==1.)[0]]

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(anger[:, 0],anger[:, 1], c='red', marker='^', label='anger')
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
        
        
def imshow(value,label, size=(50,40)):
    n = len(value)
    h =-(-n//10)
    fig = plt.figure(figsize=size)
    for i in range(n):
        plt.subplot(h, 10, (i+1))
        plt.title((str(i+1)+' Label: '+emos_inv[int(label[i])]))
        plt.imshow(value[i],cmap='gray')
    plt.show()

    
class Finetuning():
    def __init__(self,X_train, X_test, y_train, y_test,latent_size=100,class_num=6,lr=1e-4):
        self.configure()
        self.emos_inv = {1:'anger',2:'contempt',3:'disgust',4:'fear',5:'happy',6:'sad',7:'surprise'}
        self.logs_path = "../logs"
        self.weight_path_ext = '../weigths/'+'DCGAN' + '.ckpt'
        self.weight_path_cls = '../weigths/'+'finetuning' + '.ckpt'
        self.x = tf.placeholder(tf.float32, [None, X_train.shape[1]], name='InputData')
        self.Training = tf.placeholder(dtype=tf.bool, name='Training')
        self.label = tf.placeholder(tf.int32, [None, 1], name='label')
        self.vectors_train, self.vectors_test = X_train, X_test
        self.y_train, self.y_test = y_train.reshape(len(y_train),1), y_test.reshape(len(y_test),1)
        self.class_layer, self.z, self.loss, self.optimizer = self.Class_layer(self.x, self.label, class_num, latent_size,lr)
        self.class_layer_val, self.z_val, self.loss_val = self.Class_layer(self.x, self.label, class_num, latent_size,lr,reuse=True)
        
        
    def Class_layer(self,flat,y,class_num,latent_size,lr,reuse=False):
        with tf.variable_scope("Class_layer") as scope:
            if (reuse):
                scope.reuse_variables()
            z = fullyConnected(flat, name='z_FT', output_size=latent_size, activation = 'relu')
            class_layer = fullyConnected(z, name='classifier', output_size=class_num, activation = 'linear')
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=class_layer)
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            if(reuse):
                return class_layer, z, loss,
            else:
                optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
                return class_layer, z, loss, optimizer 
 
    def configure(self, epochs=20, batch_size=10,test_size=0.3):
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self):
        epochs=self.epochs 
        batch_size=self.batch_size
        with tf.name_scope('training_ft'):
            tf.summary.scalar('loss', self.loss)
        with tf.name_scope('validation_ft'):
            tf.summary.scalar('loss', self.loss_val)
        trn_summary = tf.summary.merge_all(scope='training_ft')
        val_summary = tf.summary.merge_all(scope='validation_ft')
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess_tra:
            if tf.gfile.Exists(self.logs_path):
                tf.gfile.DeleteRecursively(self.logs_path)
            file_writer = tf.summary.FileWriter(self.logs_path, sess_tra.graph)
            sess_tra.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            start_time = time.time()
            epoch_time =0.
            k = 1.
            n_train=len(self.vectors_train)
            n_test=len(self.vectors_test)
            accuracys = []
            for epoch in range(epochs):
                epoch_start = time.time()
                sum_loss = 0
                n_batches = int(n_train / batch_size)
                # Loop over all batches
                counter = 0
                perm = np.random.permutation(n_train)
                train_data = self.vectors_train[perm]
                train_y = self.y_train[perm]
                for i in range(n_batches):
                    batch_x = train_data[i*batch_size:(i+1)*batch_size]
                    batch_y = train_y[i*batch_size:(i+1)*batch_size]

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _,res_trn,classes, z ,train_cost = sess_tra.run([self.optimizer,trn_summary , self.class_layer, self.z, self.loss], feed_dict={self.x: batch_x, self.label:batch_y})
                    sum_loss += train_cost 
                    sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}, time: {}".format(counter+1, np.int(n_train/batch_size)+1, sum_loss/(i+1),(time.time()-start_time)))
                    sys.stdout.flush()
                    counter +=1
                file_writer.add_summary(res_trn, (epoch+1))
            
                saver.save(sess_tra, self.weight_path_cls)
                print('')
                print('Epoch', epoch+1, ' / ', epochs, 'Training Loss:', sum_loss/n_batches)

                res_val,classes_val, z_val ,test_cost = sess_tra.run([val_summary , self.class_layer_val, self.z_val, self.loss_val], feed_dict={self.x: self.vectors_test, self.label:self.y_test})
                print('Validation Loss:', test_cost/n_test)
                accuracy = (np.argmax(classes_val,axis=1)==self.y_test[:,0].astype(np.int)).sum()/len(self.y_test)
                accuracys += [accuracy]
                print('Accuracy:', accuracy)
                file_writer.add_summary( res_val, (epoch+1))
                epoch_end = time.time()-epoch_start
                epoch_time+=epoch_end
                print('Time per epoch: ',(epoch_time/k),'s/epoch')
                print('')
                k+=1.
        print('Optimization Finished with time: ',(time.time()-start_time))
        sess_tra.close()
        self.classes_val = classes_val
        return classes_val, z_val ,test_cost,accuracys
    
    def predict(self):
        pred = np.argmax(self.classes_val,axis=1)
        return pred,(pred.astype(np.int32)==self.y_test[:,0]).sum()/len(pred)

def mean_and_cov(X,Y):
    Mu = []
    Z = []
    y = np.unique(Y)
    for i in y:
        idx = np.where(Y==i)[0]
        mu = np.mean(X[idx],0)
        z = np.cov(X[idx].T)
        Mu+= [mu]
        Z+= [z]
    return np.array(Mu), np.array(Z)

def KL_dive(Mu_0,Z_0,Mu_1,Z_1,tol):
    d = len(Mu_0)
    detZ0=np.linalg.det(Z_0)
    detZ1=np.linalg.det(Z_1)
    invZ1=np.linalg.inv(Z_1+np.eye(d)*tol)
    mu1_mu0 = (Mu_1 - Mu_0).reshape(d,1)
    kernel = np.dot(mu1_mu0.T,np.dot(invZ1,mu1_mu0))
    0.5*(np.log(detZ1/detZ0)-d+np.trace(np.dot(invZ1,Z_0))+kernel)
    return kernel


def KL_all_conbo(X,Y,Tol=0.00001):
    Mu,Z = mean_and_cov(X,Y)
    n = len(Mu)
    KL_values=np.empty((n,n))
    for i in range(n):
         for j in range(n):
                value = KL_dive(Mu[i],Z[i],Mu[j],Z[j],tol=Tol)
                KL_values[i,j]=value[0,0]
    return KL_values