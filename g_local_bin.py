from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn import preprocessing
from skimage import morphology
from skimage import feature

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
else: path = "Dataset/1horse"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
featureSize = 3
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

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_sob = sobel(img_blur)

    img_sob_clr = sobel(img_clr)

    b, g, r = cv2.split(img_clr)
    h, s, v = cv2.split(img_hsv)
    l, la, lb = cv2.split(img_lab)
    y, cr, cb = cv2.split(img_ycb)

    numPoints = 32
    radius = 8
    eps = 1e-7

    space = [img_gray, img_sob]
    h, w = img_gray.shape
    hists = []
    for method, img in enumerate(space):
        means, stds, skews, flattens = [], [], [], []
        entropys, energys = [], []

        lbp = feature.local_binary_pattern(img, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        means = calcMeans(hist, means)
        if method == 0:
            stds = calcStds(hist, means, stds)
            hists.append(means)
        elif method == 1:
            stds = calcStds(hist, means, stds)
            hists.append(means)
            hists.append(stds)
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
            if "list" in types or "array" in types:
                n = len(hist)
            else:
                n = 1
            realFeatureSize += n
            X[i, start:start + n] = np.copy(hist[:])
            start += n
        image[i, :, :] = img_clr
        names.append(data[i])
        image_label = int(data[i].split("-")[0])
        image_label -= 1
        count.append(image_label)
        print(f"\r{i}, {data[i]}", end="")

# Save features
print("Saving features")
pickle_out = open("local_bin.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()