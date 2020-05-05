import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
style.use("ggplot")

epoches = 10
size, dimension = 10, 2

X = np.zeros((size*2, dimension))
center = np.zeros((dimension, dimension))

X[:size, 0] = np.random.normal(0, 1, size)
X[:size, 1] = np.random.normal(0, 1, size)

X[size:, 0] = np.random.normal(10, 3, size)
X[size:, 1] = np.random.normal(10, 3, size)

plt.scatter(X[:size, 0], X[:size, 1], marker="x", c="b")
plt.scatter(X[size:, 0], X[size:, 1], marker=".", c="g")

old = copy.copy(X)

np.random.shuffle(X)

plt.ion()
plt.xlim(-5, 15)
plt.ylim(-5, 15)

for i in range(dimension):
    center[i] = X[i]

for i in range(int(epoches)):

    classification = {}
    for i in range(dimension):
        classification[i] = np.zeros(dimension+1)

    for data in X:
        dist = list(np.sqrt(np.sum((data - center) ** 2, axis=1)))
        index = dist.index(min(dist))

        for i in range(dimension + 1):
            if i == dimension:
                classification[index][i] += 1
            else:
                classification[index][i] += data[i]

    for i in range(dimension):
        if classification[i][-1] != 0:
            center[i] = np.array((classification[i] / classification[i][-1])[:dimension])

    # show()
    if 'sca' in globals():
        sca.remove()
    plt.scatter(old[:size, 0], old[:size, 1], marker="x", c="b")
    plt.scatter(old[size:, 0], old[size:, 1], marker=".", c="g")
    sca = plt.scatter(center[:,0], center[:,1], marker="o", c="r")
    plt.pause(0.5)
    plt.ioff()

plt.show()
