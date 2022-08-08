#!/usr/bin/env python3
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
# create trackbars for color change
def nothing(x):
    pass

cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
while(1):
    # Take each frame
    # Create a black image, a window

    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    _,frame = cap.read()
    # Convert BGR to HSV
    #frame = cv.blur(frame,(10,10))
    #kernel = np.ones((20,20),np.float32)/400
    #frame = cv.filter2D(frame,-1,kernel)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    color = np.uint8([[[b,g,r ]]])
    hc = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    h,s,v = hc[0,0,:]
    lower_limit = np.array([h-50,s-100,v-30])
    upper_limit = np.array([h+50,s+100,v+30])
    # Threshold the HSV image to get only blue colors

    print(h,s,v)
    mask = cv.inRange(hsv, lower_limit, upper_limit)

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #for i in range(0,len(hierarchy)):
    #    if hierarchy[i,:,:]
    print(len(contours))
    for cnt in contours:
        M = cv.moments(cnt)
        #print(M)
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cv.circle(frame,(cx,cy), 20, (0,0,255), -1)
            cv.line(frame,(cx-20,cy),(cx+20,cy),(255,0,0),1)
            cv.line(frame,(cx,cy-20),(cx,cy+20),(255,0,0),1)

    #cv.drawContours(frame, contours, -1, (0,255,0), 3)
    # Bitwise-AND mask and original image
    #res = cv.bitwise_and(frame,frame, mask= mask)
    img[:] = [b,g,r]
    cv.imshow('image',img)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    #cv.imshow('res',blur)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
