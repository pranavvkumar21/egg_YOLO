#!/usr/bin/env python3
import cv2
import numpy as np
import os
import csv

ant_dir = "./test/"
os.chdir(ant_dir)
filelist = os.listdir()

with open('labels.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for i in filelist:
        file = open(i,"r")
        lines = file.readlines()
        for j in lines:
            items = j.split()
            #print(items)
            row = [i]
            for k in range(1,len(items)):
                row.append(float(items[k]))
            writer.writerow(row )
