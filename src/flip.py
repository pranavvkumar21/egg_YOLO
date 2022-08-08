#! /usr/bin/env python3
import os
import tensorflow as tf
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
folder = "/home/mobsycho100/Desktop/Scara_Poultry/egg_dataset_resized/"


os.chdir(folder)
a = os.listdir()
for i in a:
    img = cv2.imread(i)
    img_flip = tf.image.flip_left_right(img).numpy()
    cv2.imwrite("flipped_"+i,img_flip)
