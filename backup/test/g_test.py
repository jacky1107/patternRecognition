from skimage import transform
from skimage import filters
from sklearn import preprocessing

from c_local import *

import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import pandas as pd
import numpy as np
import mahotas
import shutil
import pickle
import copy
import cv2
import os

Test = "no"
names = []
clusterSize = 10
if clusterSize == 10:
    path = "Dataset/test"
elif clusterSize == 5:
    path = "Dataset/opening"
else:
    path = "Dataset/1horse"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
featureSize = 4 * 3
image = np.zeros((dataSize, 300, 400, 3), dtype="uint8")
X = np.zeros((dataSize, featureSize))

for i in range(100,dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_clr)

    img_sob = sobel(img_gray)
    ret, binary = cv2.threshold(img_sob,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), dtype="uint8")
    binary = cv2.dilate(binary, kernel, iterations=1)
    kernel = np.ones((3,3), dtype="uint8")
    binary = cv2.erode( binary, kernel, iterations=1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    img_new = np.zeros_like(img_clr, dtype=np.int16)
    img_new = img_new - 1
    img_draw = np.copy(img_clr)
    lblareas = stats[:,cv2.CC_STAT_AREA]
    for j in range(1,len(lblareas)):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]
        temp = img_clr[y:y+h, x:x+w]
        img_new[y:y+h, x:x+w] = np.copy(temp)
    
    img_new += 1
    img_new[img_new > 255] = 255
    img_new = img_new.astype("uint8")
    cv2.imshow("d", img_new)

    img_new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_new_gray,0,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5), dtype="uint8")
    binary = cv2.dilate(binary, kernel, iterations=3)

    kernel = np.ones((5,5), dtype="uint8")
    binary = cv2.erode(binary, kernel, iterations=6)
    
    binary = binary / 255
    img_clr[...,0] = img_clr[...,0] * binary
    img_clr[...,1] = img_clr[...,1] * binary
    img_clr[...,2] = img_clr[...,2] * binary
    img_clr = img_clr.astype("uint8")

    cv2.imshow("c", img_clr)
    cv2.waitKey(0)

    # fileName = currentPath + "/Dataset/morphlogy/" + name
    # cv2.imwrite(fileName, img_new)
    print(f"\r{i}", end="")
    if i > 130: break

cv2.destroyAllWindows()





    # img_gray = cv2.medianBlur(img_gray, 3)
    # img_gray = img_gray.astype("uint16")
    # img_gray = img_gray + 1
    # img_gray = np.log(img_gray)
    # f = np.fft.fft2(img_gray)
    # fshift = np.fft.fftshift(f)

    # fshift = fshift * highPass

    # f_ishift = np.fft.ifftshift(fshift)
    # img_new = np.fft.ifft2(f_ishift)
    # img_new = np.real(np.exp(img_new)) - 1
    # temp = img_new - np.amin(img_new)
    # img_new = temp * 255 / np.amax(temp)
    # img_new = img_new.astype("uint8")
    
    # img_blur = cv2.medianBlur(img_new, 3)
    # img_sob = cv2.Canny(img_blur, 100, 255)

    # kernel = np.ones((5,5), dtype=np.uint8)
    # img_mor = cv2.morphologyEx(img_sob, cv2.MORPH_CLOSE, kernel, iterations=1)

    # img_clr = np.array(img_clr, dtype=np.int16)
    # img_clr += 1
    # img_new = np.zeros_like(img_clr, dtype=np.int16)
    # lines = cv2.HoughLinesP(img_mor, 1, np.pi/180, 30, minLineLength=60, maxLineGap=10)
    # if lines is not None:
    #     lines1 = lines[:,0,:]
    #     for x1,y1,x2,y2 in lines1[:]: 
    #         cv2.line(img_clr,(x1,y1),(x2,y2),(255,0,0),1)
    #         img_new[y1:y2,x1:x2] = np.copy(img_clr[y1:y2,x1:x2])

    # img_new[img_new > 255] = 255
    # img_new[img_new < 0] = 0
    # img_new = img_new.astype("uint8")
    # cv2.imwrite(pre2, img_mor)

    # imgray = cv2.Canny(erosion,30,100)

    # circles = cv2.HoughCircles(imgray,cv2.HOUGH_GRADIENT,1,20,
    #                             param1=50,param2=30,minRadius=20,maxRadius=40)

    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    #     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    #     print(len(circles[0,:]))


    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thres)
    # img_clr = np.array(img_clr, dtype=np.int16)
    # img_clr += 1

    # img_new = np.zeros_like(img_clr, dtype=np.int16)
    # lblareas = stats[:,cv2.CC_STAT_AREA]
    # for j in range(1, len(lblareas)):
    #     x = stats[j, cv2.CC_STAT_LEFT]
    #     y = stats[j, cv2.CC_STAT_TOP]
    #     w = stats[j, cv2.CC_STAT_WIDTH]
    #     h = stats[j, cv2.CC_STAT_HEIGHT]
    #     temp = img_clr[y:y+h, x:x+w]
    #     img_new[y:y+h, x:x+w] = np.copy(temp)
    
    # img_new[img_new > 255] = 255
    # img_new[img_new < 0] = 0
    # img_new = img_new.astype("uint8")




    # levels = 4
    # for level in range(levels):

    #     h, w = higherResoGauss.shape
    #     if h % 2 == 1: higherResoGauss = higherResoGauss[:h - 1,:]
    #     if w % 2 == 1: higherResoGauss = higherResoGauss[:,:w - 1]
    #     h, w = higherResoGauss.shape
    #     ch,cw= h // 2, w // 2

    #     higherResoGauss = cv2.GaussianBlur(higherResoGauss, (5,5), 2)
    #     lowerResoGauss = cv2.resize(higherResoGauss, (cw,ch))
    #     lowerResoGauss = cv2.pyrDown(higherResoGauss)
    #     temp = cv2.pyrUp(lowerResoGauss)
    
    #     lowerResoLap = higherResoGauss - temp
    #     cv2.imwrite(pre[level], lowerResoLap)

    #     higherResoGauss = lowerResoGauss
