import cv2
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt


# sobelX = np.array([
#     [2,1,0,-1,-2],
#     [4,2,0,-2,-4],
#     [6,4,0,-4,-6],
#     [4,2,0,-2,-4],
#     [2,1,0,-1,-2]
# ])

# print(sobelX.T)

# X = pickle.load(open("features.pickle", "rb"))

# epoches = 1e+3
# n_cluster = 3
# size, dimension = X.shape
# center = np.zeros((n_cluster, dimension))

# for i in range(n_cluster):
#     index = np.random.choice(size)
#     center[i] = X[index]

# for i in range(int(epoches)):

#     classification = {}
#     for i in range(n_cluster):
#         classification[i] = [0, np.zeros((size, dimension))]

#     for data in X:
#         dist = list(np.sqrt(np.sum((data - center) ** 2, axis=1)))
#         index = dist.index(min(dist))

#         count = classification[index][0]
#         classification[index][1][count] = data
#         classification[index][0] += 1
        
#     preCenter = np.copy(center)
#     for i in range(n_cluster):
#         count = classification[i][0]
#         if count != 0:
#             data = classification[i][1]
#             summation = np.sum(data, axis=0)
#             mean = summation / count
#             center[i] = mean
    
#     optimizerA = np.sqrt(preCenter**2)
#     optimizerB = np.sqrt(center**2)
#     if np.sum(optimizerA - optimizerB) < 1e-4: break

# for i in range(n_cluster):
#     count = classification[i][0]
#     print(count)