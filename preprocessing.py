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

bag = np.zeros((1,128))
for i in range(dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    if "DS" in imagePath: os.remove(imagePath)
    img_clr = cv2.imread(imagePath)
    img_gray1 = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    img_gray1 = cv2.medianBlur(img_gray1, 5)
    h, w = img_gray1.shape

    pre1 = currentPath + "/Dataset/pre1/" + name

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1, None)
    sizeDes1 = len(des1)

    bag = np.concatenate((bag, des1), axis=0)

    print(bag.shape)
    print(f"\r{i}", end="")

bag = bag[1:]

# Save features
print("Saving features")
pickle_out = open("bag.pickle", "wb")
pickle.dump(bag, pickle_out)
pickle_out.close()


# bag = np.zeros(716803)
#     for j in range(dataSize):
#         if i == j: continue
#         name = data[j]
#         imagePath = os.path.join(dataPath, name)
#         if "DS" in imagePath: os.remove(imagePath)
#         img_clr = cv2.imread(imagePath)
#         img_gray2 = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
#         img_new = np.copy(img_clr)
#         h, w = img_gray2.shape

#         sift = cv2.xfeatures2d.SIFT_create()
#         kp2, des2 = sift.detectAndCompute(img_gray2, None)

#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(des1,des2,k=2)

#         # Apply ratio test
#         index = 0
#         for m, n in matches:
#             if m.distance < 0.75*n.distance:
                
#             index += 1


        # img_new = cv2.drawMatchesKnn(img_gray1,kp1,img_gray2,kp2,good,img_new,flags=2)
        # cv2.imwrite(pre1, img_new)