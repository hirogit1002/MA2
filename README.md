# Unsuprvised deep learning for face expression recognition v0.1.0.
Deep learning for feature extraction of image data. This program learns how to encode feature vector from image and decode image from the feature vector.

## Requirements
-Python3
-numpy
-pandas
-pickle
-scipy
-gzip
-Tensorflow 1.12.0
-OpenCV 3.4

## Installation
	$ git clone https://github.com/hirogit1002/MA2

## Quickstart Start
Move to the main directory
	$ cd /MA2
### Training
Put the training Image data into data folder. Image data has to fulfill following requirement.
-64x64 size
-JPEG file
-Croped by only face area

Then move to program folder and run train.py.
	$ cd /MA2
	$ python3 run.py

### Test
If you have already trained and simply will run the neural network, put the test Image data, which fulfills same requirement as training data, into data-test folder. And then
	$ cd /MA2
	$ python3 run.py -v 1

## Run.py command line options
-'--train','-t', default=1, type=int,help='1: Do train, 0: Do not train')
-'--test','-v', default=0, type=int)
-'--test_size','-ts', default=10, type=int)
-'--batch_size','-b', default=10, type=int)
-'--model','-m', default= "AE", type=str)
-'--init','-i', default= 1, type=int)
-'--norm','-n', default= 1, type=int)
-'--latent','-l', default= 100, type=int)
-'--epochs','-e', default= 100, type=int)