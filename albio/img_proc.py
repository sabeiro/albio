import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import cv2
import io

def metaImg(img):
    colorL = ['red','blue','green']
    r, g, b = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    bright = np.mean([r,g,b])
    color = colorL[np.argmax([r,g,b])]
    metaD = {"red":r,"blue":b,"green":g,"bright":bright,"color":color}
    return metaD

def imgDec(img):
    colorV = [(255,0,0),(0,255,0),(0,0,255)]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cannyL, grayL = img.copy(), img.copy()
    thres, colorT = np.zeros(img.shape), np.zeros(img.shape)
    for j in range(3):
        grayL[:,:,j] = gray
        for k in [50,100,150,200]:
            colV = np.zeros(3)
            colV[j] = k
            thres[:,:,j] = (img[:,:,j] > k)*k
            colT = np.array(thres[:,:,j],dtype = np.uint8)
            contours, hierarchy = cv2.findContours(colT,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image=colorT,contours=contours,contourIdx=-1,color=colV,thickness=1,lineType=cv2.LINE_AA)
        canny = cv2.Canny(colT,140,255)
        cannyL[:,:,j] = canny
    cannyL = np.array(cannyL,dtype = np.uint8)
    grayL = np.array(grayL,dtype = np.uint8)
    #cv2.watershed(road, marker_image_copy)
    metaD = metaImg(img)
    return [thres, colorT, cannyL, grayL], metaD

def watershed(img):
    ### to debug - skimage is not loading
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from scipy import ndimage
    import imutils
    op1 = cv2.pyrMeanShiftFiltering(img, 21, 51)
    gray = cv2.cvtColor(op1, cv2.COLOR_BGR2GRAY)
    thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresholded_image", thres)
    EDT = ndimage.distance_transform_edt(thres)
    localMax = peak_local_max(EDT, indices=False, min_distance=20, labels=thres)
    imagemarkers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    resultinglabels = watershed(-EDT, imagemarkers, mask=imagethreshold)
    for eachlabel in np.unique(resultinglabels):
        if eachlabel == 0:
            continue
    objectmask = np.zeros(imagegray.shape, dtype="uint8")
    objectmask[resultinglabels == eachlabel] = 255
    objects = cv2.findContours(objectmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = imutils.grab_contours(objects)
    result = max(objects, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(result)
    cv2.circle(imageread, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(imageread, "#{}".format(eachlabel), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Output_image", imageread)
    cv2.waitKey(0)
