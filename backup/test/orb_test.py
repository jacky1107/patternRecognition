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
    path = "Dataset/image"
elif clusterSize == 5:
    path = "Dataset/opening"
else:
    path = "Dataset/1horse"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
featureSize = 0
image = np.zeros((dataSize, 300, 400, 3), dtype="uint8")
X = np.zeros((dataSize, featureSize))

for i in range(50,dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_clr)

    orb = cv2.ORB_create()
    kps, des = orb.detectAndCompute(img_gray, None)

    print(des.shape)
    print(des)

    img_new = img_clr.copy()

    for marker in kps:
        cv2.drawMarker(img_clr, tuple(int(i) for i in marker.pt), (0,255,0))
    cv2.imshow("new", img_new)
    cv2.imshow("aa", img_clr)
    cv2.waitKey(0)

    print(f"\r{i}", end="")
    if i > 100: break
