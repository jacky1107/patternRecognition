from skimage.morphology import disk
from skimage.filters import rank

from c_local import *

import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import numpy as np
import shutil
import pickle
import copy
import cv2
import os

Test = "no"
names = []
clusterSize = 10
if clusterSize == 10: path = "Dataset/image"
elif clusterSize == 5: path = "Dataset/example"
else: path = "Dataset/10flower"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
featureSize = 7
image = np.zeros((dataSize, 300, 400, 3), dtype="uint8")
X = np.zeros((dataSize, featureSize))

highPass, laplaican = highPassFT(D0=30)

def normalize(image):
    temp = image - np.amin(image)
    img_new = temp * 255 / np.amax(temp)
    return img_new


def process_image(data):
    name = data
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    pre1 = currentPath + "/Dataset/pre1/" + name
    pre2 = currentPath + "/Dataset/pre2/" + name
    pre3 = currentPath + "/Dataset/pre3/" + name
    pre4 = currentPath + "/Dataset/pre4/" + name

    sobelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    sobely = np.array([[ 1, 2, 1],[ 0, 0, 0],[-1,-2,-1]])
    sobelx = np.fft.fft2(sobelx, [h, w])
    sobelx = np.fft.fftshift(sobelx)
    sobely = np.fft.fft2(sobely, [h, w])
    sobely = np.fft.fftshift(sobely)

    img_gray = img_gray.astype("float64")
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
    img_gray = np.fft.fft2(img_gray, [h, w])
    img_gray = np.fft.fftshift(img_gray)

    sobelx = img_gray * sobelx
    sobely = img_gray * sobely

    f_ishift = np.fft.ifftshift(sobelx)
    sobelx = np.fft.ifft2(f_ishift)
    f_ishift = np.fft.ifftshift(sobely)
    sobely = np.fft.ifft2(f_ishift)

    sobel = np.abs(sobelx * sobely)
    sobelx = np.abs(sobelx ** 2)
    sobely = np.abs(sobely ** 2)

    img_new = sobelx * sobely - sobel ** 2 - 0.3 * ( sobelx + sobely ) ** 2

    sobel = normalize(sobel)
    sobelx = normalize(sobelx)
    sobely = normalize(sobely)
    img_new = normalize(img_new)
    
    img_new[img_new < 250] = 0
    img_new[img_new >=250] = 255

    cv2.imwrite(pre1, sobel)
    cv2.imwrite(pre2, sobelx)
    cv2.imwrite(pre3, sobely)
    cv2.imwrite(pre4, img_new)

    hists = []
    return hists, img_clr

Test = "train"
with concurrent.futures.ProcessPoolExecutor() as executor:
    for i, info in enumerate(executor.map(process_image, data)):
        start = 0
        hists, img_clr = info
        print(f"\r{i}, {data[i]}", end="")
        if Test != "train": break


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

