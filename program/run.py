import argparse
from model import*
from train import*
from test import*
from train_dcgan import*


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
parser.add_argument('--learning_late','-r', default= 0.001, type=float)
args = parser.parse_args()

paths = np.array(glob.glob("../data/*.jpg"))
paths_test = np.array(glob.glob("../data_test/*.jpg"))
Train = args.train
Test = args.test
latent = args.latent
lr = args.learning_late

if(Train*Test):
    Train = 0
    Test =1 
    
if((1-args.init) or Test):
    print('Load latent setting')
    latent_name = '../save/'+args.model+'_latent_setting.pickle'
    with open(latent_name, 'rb') as f:
        latent = pickle.load(f)

if(Train):
    print('Train start')
    if(args.init):
        print('Save latent setting')
        latent_name = '../save/'+args.model+'_latent_setting.pickle'
        with open(latent_name, 'wb') as f:
            pickle.dump(latent, f)
    if(args.model=='DCGAN'):
        train_network_gan(paths, args.test_size, args.batch_size,args.init,latent, args.norm, args.epochs,"../logs", lr)
    else:
        train_network(paths, args.test_size, args.batch_size,args.init,latent, args.norm, args.epochs, args.model,"../logs",lr)

if(Test):
    print('Validation start')
    test_network(paths_test, latent, args.norm,[-1, 64, 64, 1], args.model,lr)