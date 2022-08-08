#!/usr/bin/env python3

import cv2
import imutils
import PIL
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import pyfiglet
#import progressbar
from skimage.io import imread
import time
import tensorflow as tf
import numpy as np
import keras
IMAGEFOLDER_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/egg_dataset/"
RESIZEFOLDER_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/dataset/"
LABEL_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/labels/"
SIZE = (400,400,3)
BATCH_SIZE = 2
split_size = 13
box_size = SIZE[0]//split_size


def get_localvalue(truth,box_size,image_size):
    x,y = truth[:,1],truth[:,2]
    column = (x*image_size).astype(int)// box_size
    local_x = (x*image_size).astype(int) % box_size
    row = (y*image_size).astype(int) // box_size
    local_y = (y*image_size).astype(int) % box_size

    return row,column,local_x,local_y

def convert_to_float(filename):
    file = open(LABEL_PATH+filename,"r")
    lines = file.readlines()
    lines_array = np.zeros((len(lines),5))
    lines = [i.replace("\n","") for i in lines]
    for i in range(0,len(lines)):
        line = lines[i].split()
        line = [float(j) for j in line]
        lines_array[i,:] = line
    return lines_array

def filelists(RESIZEFOLDER_PATH,LABEL_PATH):
    pwd = os.getcwd()
    os.chdir(RESIZEFOLDER_PATH)
    X_list = os.listdir()
    os.chdir(LABEL_PATH)
    Y_list = [(X_list[i].split(".")[0] +".txt") for i in range(len(X_list))]
    os.chdir(pwd)
    #for i in range(len(X_list)):
    #    print(X_list[i]+"     "+Y_list[i])
    return X_list,Y_list
def get_data(batch_x,batch_y,split_size,box_size,image_size):
    X = np.array([imread(RESIZEFOLDER_PATH + file_name)for file_name in batch_x])/255.0
    Y = np.zeros((len(batch_y),split_size,split_size,5))
    for i in range(0,len(batch_y)):
        ground_truths = convert_to_float(batch_y[i])
        #print(ground_truths)
        row,column,local_x,local_y = get_localvalue(ground_truths,box_size ,image_size)
        for j in range(0,len(ground_truths)):
            Y[i,row[j],column[j]] = ground_truths[j,0],local_x[j],local_y[j],ground_truths[j,3],ground_truths[j,4]

    return X,Y

def savefile(X,Y):
    np.save("X.npy",X)
    np.save("Y.npy",Y)
    print("file saved")
#
#print(X.shape)
#print(Y.shape)
