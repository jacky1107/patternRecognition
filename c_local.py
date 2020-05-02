import cv2
import numpy as np


def highPassFT(D0=100):
    n = 2
    c = 0.01
    rh, rl = 2, 0.5
    h, w = 300, 400
    ch, cw = int(h/2), int(w/2)
    highPass = np.zeros((h, w))
    laplaican = np.zeros((h, w))
    for u in range(h):
        for v in range(w):
            dist = ((u - ch) ** 2 + (v - cw) ** 2) ** 0.5
            highPass[u, v] = 1 - np.exp( -(dist ** 2) / (2 * D0 ** 2) )
            laplaican[u, v] = -4 * (np.pi ** 2) * (dist ** 2)
            # (rh - rl) * np.exp( -c * (dist ** 2) / (D0 ** 2) ) + rl
            # 1 - np.exp( -(dist ** 2) / (2 * D0 ** 2) ) 
            # 1 / ((1 + (D0 / dist)) ** (2*n))
    return highPass, laplaican

def localHE(image, blockSize=3):
    h, w = image.shape
    img_new = np.copy(image)
    step = int(blockSize/2)
    area = blockSize * blockSize
    temp = np.zeros((h + 2 * step, w + 2 * step))
    temp[step:h+step, step:w+step] = image
    image = np.copy(temp)
    for y in range(step, h - step):
        for x in range(step, w - step):
            rank = np.sum(image[y-step:y+step+1,x-step:x+step+1] > image[y, x])
            img_new[y-step, x-step] = rank * 255 / area
    img_new = img_new.astype("uint8")
    return img_new


def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


def historgram(image):
    img_hist = np.zeros(256)
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            img_hist[image[y, x]] += 1
    lst = []
    s = 0
    for j in range(len(img_hist)):
        s += img_hist[j]
        lst.append(s)
    lst = np.array(lst)
    lst = lst - np.min(lst)
    lst = lst / np.max(lst)
    lst = lst * 255
    lst = np.round(lst)
    lst = lst.astype("uint8")
    return lst