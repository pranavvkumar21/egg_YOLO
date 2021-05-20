#!/usr/bin/env python3
import cv2
import imutils
import PIL
import sys, os
import pyfiglet
import progressbar
from skimage.io import imread
import time
import tensorflow as tf
import numpy as np
import keras

IMAGEFOLDER_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/egg_dataset/"
RESIZEFOLDER_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/egg_dataset_resized/"
LABEL_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/test/"
SIZE = (400,400,3)
BATCH_SIZE = 2
SPLIT_SIZE = 13

def getlist(filepath):
    pwd = os.getcwd()
    os.chdir(filepath)
    filelist = os.listdir()
    os.chdir(pwd)
    return filelist

def filtering(box_output,threshold):
    box_confidence = box_output[:,:,:,0]
    filter = (box_confidence> threshold)
    boxes = tf.boolean_mask(box_output,filter)
    scores = tf.boolean_mask(box_confidence,filter)
    return boxes,scores,filter


def converttosize(IMAGEFOLDER_PATH,RESIZEFOLDER_PATH,SIZE):
    count = 0
    if not os.path.exists(RESIZEFOLDER_PATH):
        os.mkdir(RESIZEFOLDER_PATH)
    filelist = getlist(IMAGEFOLDER_PATH)
    for i in filelist:
        img = cv2.imread(IMAGEFOLDER_PATH+i)
        if img.shape[0]>img.shape[1]:


            perc_width = SIZE[1]/img.shape[1]
            resize_height = int(perc_width*img.shape[0])
            dim = (SIZE[1],resize_height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        if img.shape[1]>=img.shape[0]:


            perc_width = SIZE[0]/img.shape[0]
            resize_width = int(perc_width*img.shape[1])
            dim = (resize_width,SIZE[0])
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        h,w,c = img.shape
        img = img[(h//2-SIZE[0]//2):(h//2+SIZE[0]//2), (w//2-SIZE[1]//2):(w//2+SIZE[1]//2) ]
        print("number of photos processed = "+ str(count), end="\r\r")
        cv2.imwrite(RESIZEFOLDER_PATH+i,img)
        count+=1


def nms(boxes,scores ,iou_threshold, max_boxes):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes,score,max_boxes_tensor,iou_threshold)
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)

class Yolo_generator(keras.utils.Sequence) :

    def getlist(self,filepath):
        pwd = os.getcwd()
        os.chdir(filepath)
        filelist = os.listdir()
        os.chdir(pwd)
        return filelist

    def __init__(self, image_dir, label_dir,split_size,image_size, batch_size) :
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = self.getlist(self.image_dir)
        self.labels = self.getlist(self.label_dir)
        self.batch_size = batch_size
        self.split_size = split_size
        self.image_size = image_size
        self.box_size = self.image_size // self.split_size


    def __len__(self) :
        # print((np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int))
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def get_localvalue(self,truth):
        x,y = truth[:,1],truth[:,2]
        column = (x*self.image_size).astype(int)// self.box_size
        local_x = (x*self.image_size).astype(int) % self.box_size
        row = (y*self.image_size).astype(int) // self.box_size
        local_y = (y*self.image_size).astype(int) % self.box_size

        return row,column,local_x,local_y

    def convert_to_float(self,filename):
        file = open(LABEL_PATH+filename,"r")
        lines = file.readlines()
        lines_array = np.zeros((len(lines),5))
        lines = [i.replace("\n","") for i in lines]
        for i in range(0,len(lines)):
            line = lines[i].split()
            line = [float(j) for j in line]
            lines_array[i,:] = line
        return lines_array


    def get_data(self,batch_x,batch_y):
        X = np.array([imread(RESIZEFOLDER_PATH + file_name)for file_name in batch_x])/255.0
        Y = np.zeros((len(batch_y),self.split_size,self.split_size,5))
        for i in range(0,len(batch_y)):
            ground_truths = self.convert_to_float(batch_y[i])
            #print(ground_truths)
            row,column,local_x,local_y = self.get_localvalue(ground_truths)
            for j in range(0,len(ground_truths)):
                Y[i,row[j],column[j]] = ground_truths[j,0],local_x[j],local_y[j],ground_truths[j,3],ground_truths[j,4]

        return X,Y

    def __getitem__(self, idx) :
        # print("hi")
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        X,Y = self.get_data(batch_x,batch_y)
        return X,Y


def blank_labels(X_path,Y_path):
    os.chdir(X_path)
    list = os.listdir()
    os.chdir(Y_path)
    b= [i.split('.') for i in list]
    c= [b[i][0] for i in range(0,len(b))]
    for i in c:
        os.system("touch "+i+".txt")
