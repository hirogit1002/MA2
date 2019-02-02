# Unsuprvised deep learning for face expression recognition v0.1.0.
Deep learning for feature extraction of image data. This program learns how to encode feature vector from image and decode image from the feature vector.

## Requirements
    -Python3
    -numpy 1.15.4
    -pandas 0.23.4
    -Tensorflow 1.12.0
    -OpenCV 3.4
    -Pillow 5.1.0

## Installation
    $ git clone https://github.com/hirogit1002/MA2

## Quickstart 
Move to the main directory.

    $ cd /MA2
    
### Training
Put the training Image data into data folder. Image data has to fulfill following requirement.

    -64x64 size
    -JPEG file
    -Cropped by only face area

Then move to program folder and run train.py.

    $ cd /MA2
    $ python3 run.py

### Test
If you have already trained and simply will run the neural network, put the test Image data, which fulfills same requirement as training data, into data-test folder. And then

    $ cd /MA2
    $ python3 run.py -v 1

## Run.py command line options
    -('--train','-t', default=1, type=int, help='1: Do train, 0: Do not train')
    -('--test','-v', default=0, type=int, help='1: Do validate, 0: Do not validate')
    -('--test_size','-ts', default=10, type=int, help='Number of test data during training')
    -('--batch_size','-b', default=10, type=int, help='Setting batch size')
    -('--model','-m', default= "AE", type=str, help='Choosing model type. AE:Autoencoder, VAE: Variational Autoencoder')
    -('--init','-i', default= 1, type=int, help='1: init weight, 0: do not weight')
    -('--norm','-n', default= 1, type=int, help='1: normalize data, 0: do not normalize data')
    -('--latent','-l', default= 100, type=int, help='Size of latent variable(Dimension of feature vector)')
    -('--epochs','-e', default= 100, type=int, help='Choose how many epochs)')
    
## Folders description
In the MA2 folder there are following 6 folders.

    -data: User puts traing data into this folder
    -data_test: User puts test data into this folder
    -logs: Log files for tensorboard will be contained 
    -program: Main programs are here
    -save: Reconstructed image and latent variables will be saved as .pickle file after running the Test.
    -weights: Trained weights are saved here.
    