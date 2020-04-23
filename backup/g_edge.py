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
elif clusterSize == 3: path = "Dataset/10flower"
else: path = "Dataset/1horse"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
featureSize = 1200 * 4
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

    b, g, r = cv2.split(img_clr)
    h, s, v = cv2.split(img_hsv)
    l, la, lb = cv2.split(img_lab)
    y, cr, cb = cv2.split(img_ycb)

    hists = []
    Asm = []
    Con = []
    Idm = []
    Eng = []
    new = calcCovolution(img_sob, currentPath, 10, 10)
    for i in range(len(new)):
        img = new[i]
        glcm0 = getGLCM(img, 1, 0)
        glcm1 = getGLCM(img, 0, 1)
        glcm2 = getGLCM(img, 1, 1)
        glcm3 = getGLCM(img,-1, 1)
        feature0 = feature_computer(glcm0)
        feature1 = feature_computer(glcm1)
        feature2 = feature_computer(glcm2)
        feature3 = feature_computer(glcm3)
        Asm.append((feature0[0] + feature1[0] + feature2[0] + feature3[0]) / 4)
        Con.append((feature0[1] + feature1[1] + feature2[1] + feature3[1]) / 4)
        Idm.append((feature0[2] + feature1[2] + feature2[2] + feature3[2]) / 4)
        Eng.append((feature0[3] + feature1[3] + feature2[3] + feature3[3]) / 4)
        
    hists.append(Asm)
    hists.append(Con)
    hists.append(Idm)
    hists.append(Eng)
    return hists, img_clr

Test = "train"
with concurrent.futures.ProcessPoolExecutor() as executor:
    for i, info in enumerate(executor.map(process_image, data)):
        start = 0
        hists, img_clr = info
        realFeatureSize = 0
        for hist in hists:
            types = str(type(hist))
            if "list" in types or "array" in types: n = len(hist)
            else: n = 1
            realFeatureSize += n
            X[i, start:start + n] = np.copy(hist[:])
            start += n
        print(f"\r{i}, {data[i]}", end="")
        if Test != "train": break

print(realFeatureSize)
if Test == "train":
    # Save features
    print("Saving features")
    pickle_out = open("edge2.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
else:
    cv2.destroyAllWindows()