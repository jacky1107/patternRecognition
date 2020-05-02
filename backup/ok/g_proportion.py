from skimage import transform
from skimage import filters
from sklearn import preprocessing

from c_function import *

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
    path = "Dataset/example"
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

for i in range(dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath:
        os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_clr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_clr, cv2.COLOR_BGR2Lab)
    img_luv = cv2.cvtColor(img_clr, cv2.COLOR_BGR2Luv)
    img_ycb = cv2.cvtColor(img_clr, cv2.COLOR_BGR2YCrCb)
    img_cst = cv2.equalizeHist(img_gray)
    img_sob = sobel(img_gray)
    img_sob_clr = sobel(img_clr)

    b, g, r = cv2.split(img_clr)
    h, s, v = cv2.split(img_hsv)
    l, la, lb = cv2.split(img_lab)
    ll, lu, lv = cv2.split(img_luv)
    y, cr, cb = cv2.split(img_ycb)

    hists = []

    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    b_hist = b_hist.reshape(-1)
    b_hist = b_hist + 1

    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    g_hist = g_hist.reshape(-1)
    g_hist = g_hist + 1

    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    r_hist = r_hist.reshape(-1)
    r_hist = r_hist + 1

    new_bg = b_hist / g_hist
    new_br = b_hist / r_hist
    new_gr = g_hist / r_hist
    new_ab = la_hist / lb_hist

    space = [new_bg, new_br, new_gr]

    for img_hist in space:
        means, stds, skews = [], [], []
        entropys, energys = [], []

        means = calcMeans(img_hist, means)
        stds = calcStds(img_hist, means, stds)
        skews = calcSkews(img_hist, means, stds, skews)
        entropys = calcEntropys(img_hist, entropys)

        hists.append(means)
        hists.append(stds)
        hists.append(skews)
        hists.append(entropys)

    start = 0
    for hist in hists:
        n = len(hist)
        end = start + n
        X[i,start:end] = np.copy(hist)
        start = end
    print(f"\r{i}", end="")

# Save features
print("Saving features")
pickle_out = open("proportion.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

zscore = preprocessing.StandardScaler()
X = zscore.fit_transform(X)

excel = pd.DataFrame(X)
writer = pd.ExcelWriter('data.xlsx')
excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()

