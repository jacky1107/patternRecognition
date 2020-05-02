from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn import preprocessing
from skimage import morphology

from c_config import *
from c_operation import *

import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
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
if clusterSize == 10: path = "Dataset/image"
elif clusterSize == 5: path = "Dataset/example"
else: path = "Dataset/10flower"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
featureSize = 9
image = np.zeros((dataSize, 300, 400, 3), dtype="uint8")
X = np.zeros((dataSize, featureSize))

def process_image(data):
    name = data
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_clr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_clr, cv2.COLOR_BGR2Lab)
    img_ycb = cv2.cvtColor(img_clr, cv2.COLOR_BGR2YCrCb)
    img_cst = cv2.equalizeHist(img_gray)
    img_sob = sobel(img_gray)
    img_sob_clr = sobel(img_clr)

    space = [img_gray, img_sob]
    hists = []

    for method, img in enumerate(space):
        glcm0 = getGLCM(img, 1, 0)
        glcm1 = getGLCM(img, 0, 1)
        glcm2 = getGLCM(img, 1, 1)
        glcm3 = getGLCM(img,-1, 1)
        Asm0, Con0, Idm0, Eng0 = feature_computer(glcm0)
        Asm1, Con1, Idm1, Eng1 = feature_computer(glcm1)
        Asm2, Con2, Idm2, Eng2 = feature_computer(glcm2)
        Asm3, Con3, Idm3, Eng3 = feature_computer(glcm3)
        if method == 0:
            featureList = [Con0, Eng2, Eng3]
            hists.append(featureList)
        elif method == 1:
            featureList = [Asm0, Con0, Idm0, Eng0, Asm3, Idm3]
            hists.append(featureList)
    return hists, img_clr

count = []
Test = "train"
with concurrent.futures.ProcessPoolExecutor() as executor:
    for i, info in enumerate(executor.map(process_image, data)):
        start = 0
        hists, img_clr = info
        for hist in hists:
            types = str(type(hist))
            if "list" in types or "array" in types: n = len(hist)
            else: n = 1
            X[i, start:start + n] = np.copy(hist[:])
            start += n
        image[i,:,:] = img_clr
        names.append(data[i])
        image_label = int(data[i].split("-")[0])
        image_label -= 1
        count.append(image_label)
        print(f"\r{i}, {data[i]}", end="")
        if Test != "train": break

# Save features
print("Saving features")
pickle_out = open("glcm.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()