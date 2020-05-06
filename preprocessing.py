from skimage.morphology import disk
from skimage.filters import rank

from c_local import *
from c_function import *
from c_config import *
from c_operation import *

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

for i in range(dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    pre1 = currentPath + "/Dataset/pre1/" + name
    pre2 = currentPath + "/Dataset/pre2/" + name
    pre3 = currentPath + "/Dataset/pre3/" + name
    pre4 = currentPath + "/Dataset/pre4/" + name
    
    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    img_sob = sobel(img_gray)
    
    _, img_thres = cv2.threshold(img_sob, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((7,7), dtype=np.uint8)
    img_mor = cv2.morphologyEx(img_thres, cv2.MORPH_CLOSE, kernel, iterations=5)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_mor)
    img_clr = np.array(img_clr, dtype=np.int16)
    img_clr += 1

    img_new = np.zeros_like(img_clr, dtype=np.int16)
    lblareas = stats[:,cv2.CC_STAT_AREA]
    for j in range(1, len(lblareas)):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]
        temp = img_clr[y:y+h, x:x+w]
        img_new[y:y+h, x:x+w] = np.copy(temp)
    
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    img_new = img_new.astype("uint8")

    cv2.imwrite(pre1, img_sob)
    cv2.imwrite(pre2, img_mor)
    cv2.imwrite(pre3, img_thres)
    cv2.imwrite(pre4, img_new)
    hists = []
    print(f"\r{i}", end="")