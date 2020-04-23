import os
import cv2
import numpy as np
import pickle


path = "Dataset/Image"
currentPath = os.getcwd()
dataPath = os.path.join(currentPath, path)

imageList = os.listdir(dataPath)

color = np.zeros( (len(imageList) , 300, 400, 3 ) )
color = color.astype("uint8")

grays = np.zeros( (len(imageList) , 300, 400 ) )
grays = grays.astype("uint8")

for i, imageName in enumerate(imageList):
    imagePath = os.path.join(dataPath, imageName)
    img = cv2.imread(imagePath)
    gray = cv2.imread(imagePath, 0)
    color[i] = np.copy(img)
    grays[i] = np.copy(gray)

pickle_out = open("image.pickle","wb")
pickle.dump(color, pickle_out)
pickle_out.close()

pickle_out = open("grays.pickle","wb")
pickle.dump(grays, pickle_out)
pickle_out.close()