# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:14:21 2019

@author: jamie
"""

" Import packages "
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import os
import tensorflow as tf
#import cv2
from PIL import Image
#from scipy import ndimage
from scipy import misc
#from dnn_app_utils_v3 import *
" For resizing images "
from skimage.transform import resize
" For reading directory file names "
from os import listdir
" Keeps a consistent random seed "
np.random.seed(1)

" Initial Variables "
num_px = 64*4

def shuffleInUnison(array1, array2):
    assert len(array1) == len(array2)
    p = np.random.permutation(len(array1))
    return array1[p], array2[p]

def LoadImage(my_image, directory):
    fname = directory + my_image
    image = np.array(plt.imread(fname))
    image = resize(image, (num_px, num_px))
    #plt.imshow(image)
    #print(image.shape)
    return image

def LoadImages():
    directory0 = "testimage/negative/"
    directory1 = "testimage/positive/"
    listFiles0 = os.listdir(directory0)
    listFiles1 = os.listdir(directory1)
    listLength0 = len(listFiles0)
    listLength1 = len(listFiles1)
    imageLabels = np.zeros((listLength0 + listLength1, 1))
    imageLabels[0:listLength0] = 0
    imageLabels[listLength0 : imageLabels.shape[0]] = 1
    imageArray = np.zeros((listLength0 + listLength1, num_px, num_px, 3))  
    j = 0
    for file in listFiles0:
        image = LoadImage(file, directory0)
        imageArray[j] = image
        j = j + 1
    for file in listFiles1:
        image = LoadImage(file, directory1)
        imageArray[j] = image
        j = j + 1
    return imageArray, imageLabels


def RandomImage():
    train_x_orig = np.random.rand(209, 64, 64, 3)
    train_y = np.zeros((209, 1))
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


def createDataSets(imageArray, imageLabels, percentageTrain):
    # Shuffle the images and the labels
    imageArray, imageLabels = shuffleInUnison(imageArray, imageLabels)
    # Reshape all pixels into one dimensional vectors
    imageArray = imageArray.reshape((imageArray.shape[0], num_px*num_px*3, 1))
    # When loading the images we shuffled them so they are already random
    numTrain = int((imageArray.shape[0]) * percentageTrain)
    print(numTrain)
    print(imageArray.shape[0])
    Xtrain = imageArray[0 : numTrain ]
    Ytrain = imageLabels[0 : numTrain ]
    Xtest = imageArray[numTrain : imageArray.shape[0] ]
    Ytest = imageLabels[numTrain : imageArray.shape[0] ]
    return Xtrain, Xtest, Ytrain, Ytest

def reshapeAndPrint(image, labels, index):   #Reshapes a vector into an image and shows it
    image = image[index]
    image = image.reshape(num_px, num_px, 3)
    plt.imshow(image)
    print(" Label is Y= " + str(int(labels[index])) + " (Y=1 contains people, Y=0 doesn't)")


images, imageLabels = LoadImages()
#plt.imshow(Images[0])
#Xtest = createDataSets(Images)
#Xtest0 = Xtest[0]
#Xtest0 = Xtest0.reshape(num_px, num_px, 3)
#print(np.amax(Xtest0))   
Xtrain, Xtest, Ytrain, Ytest = createDataSets(images, imageLabels, 0.5)
reshapeAndPrint(Xtrain, Ytrain, 0)

def tensortest():
    y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')                    # Define y. Set to 39

    loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

    init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:                    # Create a session and print the output
        session.run(init)                            # Initializes the variables
        print(session.run(loss))                     # Prints the loss
