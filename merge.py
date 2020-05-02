from sklearn.cluster import KMeans
from sklearn.manifold import Isomap

from c_config import *
from c_function import *
from c_operation import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import pickle
import copy
import cv2
import os

def evaluate(labels):
    # Save Image by label
    rate = np.zeros((clusterSize, clusterSize))
    currentPath = os.getcwd()
    resultPath = os.path.join(currentPath, 'result/')
    shutil.rmtree(resultPath)
    os.mkdir(resultPath)
    for i, label in enumerate(labels):
        image_label = names[i]
        rate[label, image_label] += 1
        fileName = resultPath + str(label) + "_" + str(data[i])
        cv2.imwrite(fileName, image[i, :, :])

    # calculate recall rate and precision rate
    ch_name = ["馬兒","飛機","雞群","森林","帆船","機車","汽車","蝴蝶","蜻蜓","花兒"]
    sortedList = []
    table = np.zeros(clusterSize)
    recall = np.zeros(clusterSize)
    precision = np.zeros(clusterSize)
    for i in range(len(rate)):
        tp = max(rate[i])
        index = np.where(rate[i] == tp)[0][0]
        sortedList.append(index)
        table[index] += 1

    rate = rate[np.argsort(sortedList)]
    sortedList = sorted(sortedList)
    for i in range(len(rate)):
        index = sortedList[i]
        tp = max(rate[i])
        fp = np.sum(rate[i]) - tp
        fn = np.sum(rate[:, index]) - tp
        precisionRate = tp / (tp + fp)
        recallRate = tp / (tp + fn)
        recall[i] = recallRate
        precision[i] = precisionRate
    averagePrecision = np.mean(precision)
    averageRecall = np.mean(recall)
    return table, averagePrecision, averageRecall



Test = "no"
clusterSize = 10
if clusterSize == 10: path = "Dataset/image"
elif clusterSize == 5: path = "Dataset/example"
else: path = "Dataset/10flower"
colors = ['r', 'g', 'b', 'k', 'm']
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)
data = os.listdir(dataPath)
dataSize = len(data)
image = np.zeros((dataSize, 300, 400, 3), dtype="uint8")
names = []

for i in range(dataSize):
    name = data[i]
    imagePath = os.path.join(dataPath, name)
    img_clr = cv2.imread(imagePath)
    image[i,:,:] = img_clr
    image_label = int(data[i].split("-")[0])
    image_label -= 1
    names.append(image_label)

functions = pickle.load(open("pickle/functions.pickle", "rb"))
glcms = pickle.load(open("pickle/glcm.pickle", "rb"))
localBins = pickle.load(open("pickle/local_bin.pickle", "rb"))
proportion = pickle.load(open("pickle/proportion.pickle", "rb"))
second_order = pickle.load(open("pickle/second_order.pickle", "rb"))
roi_local_bin = pickle.load(open("pickle/roi_local_bin.pickle", "rb"))

p_glcm = pickle.load(open("p_glcm.pickle", "rb"))
p_functions = pickle.load(open("p_functions.pickle", "rb"))
p_proportion = pickle.load(open("p_proportion.pickle", "rb"))
p_second_order = pickle.load(open("p_second_order.pickle", "rb"))
p_test = pickle.load(open("test.pickle", "rb"))

features = [
    functions, functions, localBins, proportion, glcms, second_order, roi_local_bin,
]

start = 0
dimensions = []
for i in range(len(features)):
    _, dimension = features[i].shape
    dimensions.append(dimension)
featureSize = np.sum(dimensions)
X = np.zeros((dataSize, featureSize))

for i in range(len(features)):
    end = start + dimensions[i]
    X[:,start:end] = np.copy(features[i])
    start = end
    print(f"Feature{i+1}: {dimensions[i]}")
print(f"Total parameters: {featureSize}")

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)
X = (X - x_mean) / x_std

temp = X[np.argsort(names)]
excel = pd.DataFrame(temp)
writer = pd.ExcelWriter('data.xlsx')
excel.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()

# KMeans
bestRecall = 0
bestPrecision = 0
bestLabels = None
epoches = 1
for i in range(epoches):
    print(f"\rKmeans loading {round(i/epoches*100, 2)}", end="")
    clf = KMeans(init="k-means++", n_clusters=clusterSize, random_state=42)
    labels = clf.fit_predict(X)
    table, averagePrecision, averageRecall = evaluate(labels)
    best = (bestPrecision + bestRecall) / 2
    now = (averagePrecision + averageRecall) / 2
    if now > best:
        bestLabels = labels
        bestPrecision = averagePrecision
        bestRecall = averageRecall
        if 0 not in bestLabels:
            bestLabels = labels
            bestPrecision = averagePrecision
            bestRecall = averageRecall
            
labels = bestLabels
X_iso = Isomap(n_neighbors=10).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(top=0.85)
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], s=8, c=names)
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], s=8, c=labels)
print("\n")

# Save Image by label
rate = np.zeros((clusterSize, clusterSize))
currentPath = os.getcwd()
resultPath = os.path.join(currentPath, 'result/')
shutil.rmtree(resultPath)
os.mkdir(resultPath)
for i, label in enumerate(labels):
    image_label = names[i]
    rate[label, image_label] += 1
    fileName = resultPath + str(label) + "_" + str(data[i])
    cv2.imwrite(fileName, image[i, :, :])

# calculate recall rate and precision rate
ch_name = ["馬兒","飛機","雞群","森林","帆船","機車","汽車","蝴蝶","蜻蜓","花兒"]
sortedList = []
table = np.zeros(clusterSize)
recall = np.zeros(clusterSize)
precision = np.zeros(clusterSize)
for i in range(len(rate)):
    tp = max(rate[i])
    index = np.where(rate[i] == tp)[0][0]
    sortedList.append(index)
    table[index] += 1

rate = rate[np.argsort(sortedList)]
sortedList = sorted(sortedList)
for i in range(len(rate)):
    index = sortedList[i]
    tp = max(rate[i])
    fp = np.sum(rate[i]) - tp
    fn = np.sum(rate[:, index]) - tp
    precisionRate = tp / (tp + fp)
    recallRate = tp / (tp + fn)
    recall[i] = recallRate
    precision[i] = precisionRate

    print(index, ch_name[index], rate[i],
            "P", round(precisionRate, 2),
            "R", round(recallRate, 2)
            )

print(f"Average precision rate: {np.mean(precision)}")
print(f"Average recall rate: {np.mean(recall)}")
print(table)
print("\n")
plt.show()

# temp = np.copy(X)
# start = 57
# print("\n")
# for i in range(64):
#     print(start)
#     X = np.concatenate((temp[:,:57], temp[:,start:start+1]), axis=1)
#     start += 1