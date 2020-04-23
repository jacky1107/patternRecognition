import cv2
import math
import numpy as np


def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

def calcGradient(image):
    h, w, c = image.shape
    gradient = np.zeros((h, w))
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            diff = 0
            diff += abs(image[y, x] - image[y - 1, x])
            diff += abs(image[y, x] - image[y, x - 1])
            diff += abs(image[y, x] - image[y + 1, x])
            diff += abs(image[y, x] - image[y, x + 1])
            diff = np.sum(diff)
            diff = diff / 4
            gradient[y, x] = diff
    return gradient


def normalizeImage(grad):
    return (grad - np.min(grad)) * 255.0 / np.max(grad)


def findMinEnergySeam(grad):
    height, width = grad.shape
    carve = np.zeros((height, width))
    max = None
    carve[0, :] = grad[0, :]
    for y in range(1, height):
        for x in range(width):
            emin = None
            for tx, ty in [[x-1, y-1], [x, y-1], [x+1, y-1]]:
                if (tx >= 0) and (tx < width) and (ty >= 0) and (ty < height):
                    if (emin is None) or (grad[y, x] + grad[ty, tx] < emin):
                        emin = grad[y, x] + grad[ty, tx]
            carve[y, x] = emin
            if max is None or emin > max:
                max = emin
    return carve


def findPath(carve):
    height, width = carve.shape
    minIndex = None
    min = None
    for x in range(width):
        if min is None or carve[height-1, x] < min:
            minIndex = x
            min = carve[height-1, x]

    cx = minIndex
    path = [[height-1, cx]]

    for y in range(height-2, -1, -1):
        min = None
        minX = None
        for dx in [cx-1, cx, cx+1]:
            if (dx >= 0) and (dx < width):
                if min is None or carve[y, dx] < min:
                    min = carve[y, dx]
                    minX = dx
        cx = minX
        path.append([y, cx])
    return path


def removePath(img, path):
    height, width, depth = img.shape
    new = np.zeros((height, width-1, depth))
    # print(img.shape, new.shape)

    for y, x in path:
        # r = img[y,:x+1] + img[y,x+1:]
        # print('x',x)
        new[y, 0:x] = img[y, 0:x]
        new[y, x:] = img[y, x+1:]
    return new


def carveSeam(origImg, NIter):
  img = origImg.copy()
  for i in range(NIter):
    print('Loop', i, 'Shape', img.shape)
    grad = calcGradient(img)
    carve = findMinEnergySeam(grad)
    minPath = np.array(findPath(carve))
    img = removePath(img, minPath)

    img = np.rot90(img)
    grad = calcGradient(img)
    carve = findMinEnergySeam(grad)
    minPath = np.array(findPath(carve))
    
    img = removePath(img, minPath)
    img = np.rot90(img, 3)

  return img.astype(np.uint8)

# def gray2contrast(image, hist):
#     new = np.copy(image)
#     nn = []
#     n = 0
#     L = len(hist) - 1
#     m = np.sum(hist)
#     h, w = new.shape
#     for i in range(len(hist)):
#         n += float(hist[i])
#         nn.append(n)
#     nn = np.array(nn)
#     nn = nn / max(nn) * 255
#     for x in range(h):
#         for y in range(w):
#             new[x, y] = int(nn[new[x, y]])
#     return new


# def convolution(image, kernel_size=(3, 3), out_channels=1):
#     n = len(image.shape)
#     kr, kc = kernel_size[0], kernel_size[1]
#     if n > 2:
#         h, w, c = image.shape
#         masks = np.random.randint(-1, 2, (kr, kc, c))
#     else:
#         h, w = image.shape
#         masks = np.random.randint(-1, 2, (kr, kc))

#     img_cov = np.zeros((h, w))
#     for x in range(1, h - 1):
#         for y in range(1, w - 1):
#             cov = np.array([
#                 [image[x-1, y-1], image[x, y-1], image[x+1, y-1]],
#                 [image[x-1, y], image[x, y], image[x+1, y]],
#                 [image[x-1, y+1], image[x, y+1], image[x+1, y+1]]
#             ])
#             res = np.sum(cov * masks)
#             img_cov[x, y] = res
#     output = np.copy(img_cov[1:h-1, 1:w-1])
#     return output


# def max_pooling(image):
#     h, w = image.shape
#     img_max = np.zeros((int(h/2), int(w/2)))
#     for x in range(0, h, 2):
#         for y in range(0, w, 2):
#             ix, iy = int(x/2), int(y/2)
#             img_max[ix, iy] = max(
#                 image[x, y],
#                 image[x+1, y],
#                 image[x, y+1],
#                 image[x+1, y+1],
#             )
#     return img_max
