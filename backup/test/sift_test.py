from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import numpy as np
import shutil
import pickle
import copy
import cv2
import os

bag = pickle.load(open("p_sift_bag.pickle", "rb"))
print(bag.shape)

k = 450
batch_size = 10000
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(bag)

Test = "no"
names = []
clusterSize = 10
if clusterSize == 10: path = "Dataset/image"
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
image = np.zeros((dataSize, 300, 400, 3), dtype="uint8")

X = np.zeros((dataSize, k))

for i in range(dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray1 = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    img_gray1 = cv2.medianBlur(img_gray1, 5)
    h, w = img_gray1.shape

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1, None)
    sizeDes1 = len(des1)

    for des in des1:
        temp = des.reshape(1, -1)
        idx = kmeans.predict(temp)
        X[i, int(idx)] = X[i, int(idx)] + 1 / len(des1)

    print(f"\r{i}", end="")

print(X)
# Save features
print("Saving features")
pickle_out = open("bag_of_words_all.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()