import argparse
from model import*
from train import*
from test import*


parser = argparse.ArgumentParser(description='Running')
parser.add_argument('--path','-d', default="../data/*.jpg", type=str)
parser.add_argument('--path_test','-td', default="../data_test/*.jpg", type=str)
parser.add_argument('--train','-t', default=0, type=int)
parser.add_argument('--test','-v', default=1, type=int)
parser.add_argument('--test_size','-ts', default=10, type=int)
parser.add_argument('--batch_size','-b', default=10, type=int)
parser.add_argument('--model','-m', default= "AE", type=str)
parser.add_argument('--init','-i', default= 0, type=int)
parser.add_argument('--norm','-n', default= 1, type=int)
parser.add_argument('--latent','-l', default= 100, type=int)
parser.add_argument('--epochs','-e', default= 100, type=int)
args = parser.parse_args()

paths = np.array(glob.glob(args.path))
paths_test = np.array(glob.glob(args.path_test))
for i in paths_test:
    print(i)