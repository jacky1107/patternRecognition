import os
import cv2
import numpy as np


def calcCovolution(space, fh, fw, method):
    lst = []
    for img in space:
        means = []
        mean, std, skew = [], [], []
        n = len(img.shape)
        if n < 3: h, w = img.shape
        else: h, w, c = img.shape
        hi, wi = h/fh, w/fw
        stepStartH = int(fh / 2)
        stepStartW = int(fw / 2)
        for y in range(stepStartH, h - stepStartH + 1, fh):
            for x in range(stepStartW, w - stepStartW + 1, fw):
                cov = img[y-stepStartH:y+stepStartH+1, x-stepStartW:x+stepStartW+1]
                img_hist = cv2.calcHist([cov], [0], None, [256], [0, 256])
                img_hist = img_hist.reshape(-1)
                means = calcMeans(img_hist, means)
        if method == "mean":
            lst = calcMeans(means, lst)
        elif method == "std":
            mean = calcMeans(means, mean)
            lst = calcStds(means, mean, lst)
        elif method == "skew":
            mean = calcMeans(means, mean)
            std = calcStds(means, mean, std)
            lst = calcSkews(means, mean, std, lst)
        elif method == "entropy":
            lst = calcEntropys(means, lst)
    return lst


def features(space, lst, method):
    for img in space:
        means, stds, skews, smooth = [], [], [], []
        entropys, energys = [], []
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        img_hist = img_hist.reshape(-1)
        means = calcMeans(img_hist, means)
        if method == "mean":
            lst.append(means)
        if method == "std":
            stds = calcStds(img_hist, means, stds)
            lst.append(stds)
        if method == "skew":
            stds = calcStds(img_hist, means, stds)
            skews = calcSkews(img_hist, means, stds, skews)
            lst.append(skews)
        if method == "entropy":
            entropys = calcEntropys(img_hist, entropys)
            lst.append(entropys)
        if method == "r":
            stds = calcStds(img_hist, means, stds)
            smooth = calcSmooth(img_hist, stds, smooth)
            lst.append(smooth)
    return lst

def calcSmooth(hist, stds, lst):
    std = float(stds[0])
    r = 1 - ( 1 / (1 + (std ** 2)) )
    lst.append(r)
    return lst

def calcEntropys(hist, lst):
    entropy = 0
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        if pg > 0: entropy += float(pg * np.log2(pg))
    lst.append(entropy * -1)
    return lst

def calcMeans(hist, lst):
    mean = 0
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        mean += float(g * pg)
    lst.append(mean)
    return lst

def calcStds(hist, means, lst):
    std = 0
    mean = float(means[0])
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        diff = (g - mean)
        diff = diff * diff
        n = diff * pg
        std += n
    std = np.sqrt(std)
    lst.append(float(std))
    return lst


def calcSkews(hist, means, stds, lst):
    skew = 0
    mean = float(means[0])
    std = float(stds[0])
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        diff = (g - mean)
        diff = diff * diff * diff
        n = diff * pg
        skew += n
    std = np.array(std)
    std3 = std * std * std
    if std3 != 0: skew = skew / std3
    else: skew = 0
    skew = float(skew)
    lst.append(skew)
    return lst


def calcEnergys(hist, lst):
    energy = 0
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        energy += (pg ** 2)
    lst.append(float(energy))
    return lst


def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)