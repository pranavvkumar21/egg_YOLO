#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import keras
from util import *
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import model_from_json


IMAGEFOLDER_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/egg_dataset/"
RESIZEFOLDER_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/egg_dataset_resized/"
LABEL_PATH = "/home/mobsycho100/Desktop/Scara_Poultry/test/"
SIZE = (400,400,3)
BATCH_SIZE = 2
SPLIT_SIZE = 13

def YOLO_model(input_shape):
    X_input = Input(input_shape)
    #400*400
    X = Conv2D(64, (3,3), strides = (1,1),padding='same', name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = Conv2D(64, (3,3), strides = (1,1),padding='same', name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    #200*200
    X = Conv2D(128, (3,3), strides = (1,1),padding='same', name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (3,3), strides = (1,1),padding='same', name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    #100*100
    X = Conv2D(256, (3,3), strides = (1,1),padding='same', name = 'conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3,3), strides = (1,1),padding='same', name = 'conv5')(X)
    X = BatchNormalization(axis = 3, name = 'bn5')(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3,3), strides = (1,1),padding='same', name = 'conv6')(X)
    X = BatchNormalization(axis = 3, name = 'bn6')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)
    #50*50
    X = Conv2D(512, (3,3), strides = (1,1),padding='same', name = 'conv7')(X)
    X = BatchNormalization(axis = 3, name = 'bn7')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1),padding='same', name = 'conv8')(X)
    X = BatchNormalization(axis = 3, name = 'bn8')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3,3), strides = (1,1),padding='same', name = 'conv9')(X)
    X = BatchNormalization(axis = 3, name = 'bn9')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool4')(X)
    #25*25
    X = Conv2D(1024, (3,3), strides = (1,1),padding='same', name = 'conv10')(X)
    X = BatchNormalization(axis = 3, name = 'bn10')(X)
    X = Activation('relu')(X)
    X = Conv2D(5, (3,3), strides = (2,2),padding='same', name = 'conv11')(X)
    #13*13
    model = Model(inputs = X_input, outputs = X, name='egg_YOLO')
    model.summary()
    return model

def loss_function(target,prediction):
    mse = tf.keras.losses.mean_squared_error
    existbox = target[:,:,0]
    lambda_box = 5
    lambda_obj = .05
    lambda_noobj = .05
    #----box loss---------#

    box_predictions = existbox * (prediction[:,:,:,1:5])
    box_target = existbox * (target[:,:,:,1:5])
    box_predictions[:,:,2:4] = tf.math.sign(box_predictions[:,:,:,2:4])* tf.math.sqrt(tf.math.abs(box_predictions[:,:,:,2:4]+1e-6))
    box_target[:,:,:,2:4] = tf.math.sqrt(box_target[:,:,:,2:4])
    box_predictions = tf.reshape(box_predictions,[-1,4])
    box_target = tf.reshape(box_target,[-1,4])
    box_loss = mse(box_target,box_predictions)

    #-----object loss------#
    box_confidence = tf.reshape(existbox * prediction[:,:,:,0],[-1])
    target_confidence = tf.reshape(existbox * target[:,:,:,0],[-1])
    object_loss = mse(target_confidence,box_confidence)

    #-----no object loss-----#

    noobj_prediction = tf.reshape((1-existbox) * prediction[:,:,:,0],[-1])
    noobj_target = tf.reshape((1-existbox) * target[:,:,:,0],[-1])
    noobj_loss =  mse(noobj_target,noobj_prediction)

    #----total loss---#

    loss = (lambda_box * box_loss
            + lambda_obj * object_loss
            + lambda_noobj * noobj_loss
            )
    return loss

def Train():
    generator_yolo = Yolo_generator(RESIZEFOLDER_PATH,LABEL_PATH,SPLIT_SIZE,SIZE[0],BATCH_SIZE)
    opt = keras.optimizers.Adam()
    model = YOLO_model(SIZE)
    model.compile(optimizer=opt,loss=loss_function,metrics=["accuracy"])
    model.fit_generator(generator=generator_yolo,epochs = 10)
Train()
#blank_labels(RESIZEFOLDER_PATH,LABEL_PATH)
