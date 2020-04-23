import os
import cv2
import numpy as np


def histogramReduction(hist, step):
    j = 0
    start = 0
    size = len(hist)
    reductionSize = int(size / step)
    new = np.zeros(reductionSize)
    for i in range(step, size + 1, step):
        end = start + step
        new[j] = np.mean(hist[start:end])
        start = end
        j += 1
    return new


def features(space, lst, method):
    for img in space:
        means, stds, skews, flattens = [], [], [], []
        entropys, energys = [], []
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        img_hist = img_hist.reshape(-1)
        img_hist = histogramReduction(img_hist, 2)
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
    skew = skew / std3
    skew = float(skew)
    lst.append(skew)
    return lst

def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)