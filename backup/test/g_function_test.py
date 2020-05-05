from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn import preprocessing
from skimage import morphology

from c_function import *

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
featureSize = 400
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
    img_luv = cv2.cvtColor(img_clr, cv2.COLOR_BGR2Luv)
    img_ycb = cv2.cvtColor(img_clr, cv2.COLOR_BGR2YCrCb)
    img_cst = cv2.equalizeHist(img_gray)

    b, g, r = cv2.split(img_clr)
    h, s, v = cv2.split(img_hsv)
    l, la, lb = cv2.split(img_lab)
    y, cr, cb = cv2.split(img_ycb)

    space = [img_gray]

    hists = []
    for img in space:
        h, w = img.shape
        sh, sw = 15, 20
        means, stds, skews, entropys = [], [], [], []
        for y in range(sh, h, sh * 2):
            for x in range(sw, w, sw * 2):
                cov = img[y-sh:y+sh, x-sw:x+sw]
                img_hist = cv2.calcHist([cov], [0], None, [256], [0, 256])
                img_hist = img_hist.reshape(-1)
                means = calcMeans(img_hist, means)
                stds = calcStds(img_hist, means, stds)
                skews = calcSkews(img_hist, means, stds, skews)
                entropys = calcEntropys(img_hist, entropys)
        hists.append(means)
        hists.append(stds)
        hists.append(skews)
        hists.append(entropys)

    return hists, img_clr

count = []
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
        image[i,:,:] = img_clr
        names.append(data[i])
        image_label = int(data[i].split("-")[0])
        image_label -= 1
        count.append(image_label)
        print(f"\r{i}, {data[i]}", end="")
        if Test != "train": break

# Save features
print("Saving features")
pickle_out = open("p_functions.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()