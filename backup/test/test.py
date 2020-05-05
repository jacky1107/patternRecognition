from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn import preprocessing
from skimage import morphology

from config import *
from operation import *
from K_mean_algorithm import *

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
featureSize = 16 * 15 + 1 + 4 * 3 + 20
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

    meanStdSpace = [img_cst, img_sob, img_sob_clr]
    entropySpace = [img_cst, img_sob, img_sob_clr, cr, cb, la, lb, s, v]
    
    hists = []

    hists = features(meanStdSpace, hists, "mean-std")
    hists = features(entropySpace, hists, "entropys")

    # gray 8 levels
    glcm0 = getGLCM(img_gray, 1, 0)
    glcm1 = getGLCM(img_gray, 0, 1)
    glcm2 = getGLCM(img_gray, 1, 1)
    glcm3 = getGLCM(img_gray,-1, 1)
    feature0 = feature_computer(glcm0)
    feature1 = feature_computer(glcm1)
    feature2 = feature_computer(glcm2)
    feature3 = feature_computer(glcm3)
    hists.append(feature0)
    hists.append(feature1)
    hists.append(feature2)
    hists.append(feature3)

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
pickle_out = open("features.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

if Test == "train":
    size, dimension = X.shape

    X = 1 / (1 + np.exp(-X))

    # KMeans
    print(f"Total parameters: {featureSize}")
    print("Kmeans loading")
    clf = KMeans(init="k-means++", n_clusters=clusterSize, random_state=42)
    labels = clf.fit_predict(X)

    X_iso = Isomap(n_neighbors=10).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(top=0.85)
    ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=count)
    ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=labels)
    
    # Save Image by label
    rate = np.zeros((clusterSize, clusterSize))
    currentPath = os.getcwd()
    resultPath = os.path.join(currentPath, 'result/')
    shutil.rmtree(resultPath)
    os.mkdir(resultPath)
    for i, label in enumerate(labels):
        image_label = int(names[i].split("-")[0])
        image_label -= 1
        rate[label, image_label] += 1
        fileName = resultPath + str(label) + "_" + str(names[i])
        cv2.imwrite(fileName, image[i, :, :])

    # calculate recall rate and precision rate
    table = np.zeros(clusterSize)
    recall = np.zeros(clusterSize)
    precision = np.zeros(clusterSize)
    for i in range(len(rate)):
        tp = max(rate[i])
        index = np.where(rate[i] == tp)[0][0]
        table[index] += 1
        fp = np.sum(rate[i]) - tp
        fn = np.sum(rate[:, index]) - tp
        precisionRate = tp / (tp + fp)
        recallRate = tp / (tp + fn)
        recall[i] = recallRate
        precision[i] = precisionRate

        print(i, rate[i],
              "P", round(precisionRate, 2),
              "R", round(recallRate, 2)
              )

    print(f"Average precision rate: {np.mean(precision)}")
    print(f"Average recall rate: {np.mean(recall)}")
    print(table)
    plt.show()
else:
    cv2.destroyAllWindows()
