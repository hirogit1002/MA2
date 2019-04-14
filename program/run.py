import argparse
from model import*
from train import*
from test import*
from train_dcgan import*
from test_dcgan import*

parser = argparse.ArgumentParser(description='Running')
parser.add_argument('--train','-t', default=1, type=int)
parser.add_argument('--test','-v', default=0, type=int)
parser.add_argument('--test_size','-ts', default=10, type=int)
parser.add_argument('--batch_size','-b', default=10, type=int)
parser.add_argument('--model','-m', default= "AE", type=str)
parser.add_argument('--pool','-p', default= "none", type=str)
parser.add_argument('--vector_path','-vp', default= "", type=str)
parser.add_argument('--ext_typ','-et', default= "test", type=str)
parser.add_argument('--init','-i', default= 1, type=int)
parser.add_argument('--norm','-n', default= 1, type=int)
parser.add_argument('--latent','-l', default= 100, type=int)
parser.add_argument('--epochs','-e', default= 100, type=int)
parser.add_argument('--learning_late','-r', default= 1e-4, type=float)
args = parser.parse_args()

paths = np.array(sorted(glob.glob("../data/*.jpg")))
paths_test = np.array(sorted(glob.glob("../data_test/*.jpg")))
Train = args.train
Test = args.test
latent = args.latent
lr = args.learning_late
pool_typ = args.pool
vector_path = args.vector_path

if(Train*Test):
    Train = 0
    Test =1 
    

if(Train):
    print('Train start')
    if(args.model=='DCGAN'):
        train_network_gan(paths, args.test_size, args.batch_size,args.init,latent, args.norm, args.epochs,"../logs", lr)
    else:
        train_network(paths, args.test_size, args.batch_size,args.init,latent, args.norm, args.epochs, args.model,"../logs",lr)

if(Test):
    print('Validation start')
    if(args.model=='DCGAN'):
        test_network_gan(args.test_size, latent, args.norm, lr, vector_path, pool_typ,args.ext_typ)
    else:      
        test_network(paths_test, latent, args.norm, args.model,lr,args.ext_typ)
    
    
    