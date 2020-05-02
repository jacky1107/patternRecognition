import os
import cv2
import numpy as np

def moment_m(image, p, q):
    s = 0
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            s = s + (x ** p) * (y ** q) * image[y, x]
    return s

def moment_u(image, p, q, x_mean, y_mean):
    s = 0
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            s = s + ((x - x_mean) ** p) * ((y - y_mean) ** q) * image[y, x]
    return s

def moment(space, lst):

    for img in space:

        m01 = moment_m(img, 0, 1)
        m10 = moment_m(img, 1, 0)
        m00 = moment_m(img, 0, 0)

        x_mean = m10 / m00
        y_mean = m01 / m00

        u00 = moment_u(img, 0, 0, x_mean, y_mean)
        u02 = moment_u(img, 0, 2, x_mean, y_mean)
        u03 = moment_u(img, 0, 3, x_mean, y_mean)
        u11 = moment_u(img, 1, 1, x_mean, y_mean)
        u12 = moment_u(img, 1, 2, x_mean, y_mean)
        u20 = moment_u(img, 2, 0, x_mean, y_mean)
        u21 = moment_u(img, 2, 1, x_mean, y_mean)
        u30  = moment_u(img, 3, 0, x_mean, y_mean)

        r = (0 + 2) / 2 + 1
        n02 = u02 / (u00 ** r)

        r = (0 + 3) / 2 + 1
        n03 = u03 / (u00 ** r)

        r = (1 + 1) / 2 + 1
        n11 = u11 / (u00 ** r)

        r = (1 + 2) / 2 + 1
        n12 = u12 / (u00 ** r)

        r = (2 + 0) / 2 + 1
        n20 = u20 / (u00 ** r)

        r = (2 + 1) / 2 + 1
        n21 = u21 / (u00 ** r)

        r = (3 + 0) / 2 + 1
        n30 = u30 / (u00 ** r)

        o1 = n20 + n02
        o2 = (n20 - n02) ** 2 + 4 * (n11 ** 2)
        o3 = (n30 - 3*n12) ** 2 + (3*n21 - n03) ** 2
        o4 = (n30 + n12) ** 2 + (n21 + n03) ** 2

        temp1 = (n30 + n12) ** 2 - 3 * ((n21 + n03) ** 2)
        temp2 = 3*((n30 + n12) ** 2) - (n21 + n03) ** 2
        o5 = (n30 - 3*n12) * (n30 + n12) * temp1 + (3*n21 - n03) * (n21 + n03) *  temp2
        
        temp = (n30 + n12) ** 2 - (n21 + n03) ** 2
        o6 = (n20 - n02) * temp + 4 * n11 * (n30 + n12) * (n21 + n03)
        o7 = (3*n21 - n03) * (n30 + n12) * temp1 + (3*n12 - n30) * (n21 + n03) * temp2

        lst.append([o1, o2, o3, o4, o5, o6, o7])
    return lst