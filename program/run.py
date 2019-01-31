import argparse
from model import*
from train import*
from test import*


parser = argparse.ArgumentParser(description='Running')
parser.add_argument('--train','-t', default=1, type=int)
parser.add_argument('--test','-v', default=0, type=int)
parser.add_argument('--test_size','-ts', default=10, type=int)
parser.add_argument('--batch_size','-b', default=10, type=int)
parser.add_argument('--model','-m', default= "AE", type=str)
parser.add_argument('--init','-i', default= 1, type=int)
parser.add_argument('--norm','-n', default= 1, type=int)
parser.add_argument('--latent','-l', default= 100, type=int)
parser.add_argument('--epochs','-e', default= 100, type=int)
args = parser.parse_args()

paths = np.array(glob.glob("../data/*.jpg"))
paths_test = np.array(glob.glob("../data_test/*.jpg"))
Train = args.train
Test = args.test
if(Train*Test):
    Train = 0
    Test =1 

if(Train):
    print('Train start')
    y_value,out=train_network(paths, args.test_size, args.batch_size,args.init,args.latent, args.norm,[-1, 64, 64, 1], args.epochs, args.model,"../logs")

if(Test):
    print('Validation start')
    test_network(paths_test, args.latent, args.norm,[-1, 64, 64, 1], args.model)