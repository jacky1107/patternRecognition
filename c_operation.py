import cv2
import math
import numpy as np
from sklearn import preprocessing


def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


def features(space, lst, method):
    for img in space:
        flattens, skews, means, stds = [], [], [], []
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        img_hist = img_hist.reshape(-1)
        if "mean" in method:
            means = calcMeans(img_hist, means)
            lst.append(means)
        if "entropys" == method:
            entropys = calcEntropys(img_hist)
            lst.append(entropys)
    return lst


def gamma(image, gamma=2.2):
    img_gamma = np.power(image/float(np.max(image)), gamma)
    img_gamma = img_gamma.astype("uint8")
    return img_gamma


def historgramFeatures(image, img_hist, hists):
    # img_hist = normalize(img_hist, "l2")
    img_hist = img_hist.reshape(-1)
    hists.append(img_hist)
    return hists


def covImage(image, size):
    i = 0
    s = int(size/2)
    h, w = image.shape
    totalPixel = (h - s) * (w - s)
    new = np.zeros((totalPixel, size, size))
    for x in range(s, h - s, s):
        for y in range(s, w - s, s):
            cov = image[x-s:x+s+1, y-s:y+s+1]
            new[i] = cov
            i += 1
    return new


def calcEntropys(hist):
    entropys = []
    entropy = 0
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        if pg > 0:
            entropy += float(pg * np.log2(pg))
        if g % 8 == 7:
            entropys.append(entropy)
            entropy = 0
    entropys = np.array(entropys)
    entropys = entropys * -1
    return list(entropys)


def calcMeans(hist, lst):
    mean = 0
    m = np.sum(hist)
    for g in range(len(hist)):
        pg = hist[g] / m
        mean += float(g * pg)
        if g % 8 == 7:
            lst.append(mean)
            mean = 0
    return lst


def calcGradientSobelEdge(image):
    h, w = image.shape
    sobelX = np.array([
        [2, 1, 0, -1, -2],
        [4, 2, 0, -2, -4],
        [6, 4, 0, -4, -6],
        [4, 2, 0, -2, -4],
        [2, 1, 0, -1, -2]
    ])
    sobelY = sobelX.T
    fh, fw = sobelX.shape
    step = int(fh / 2)
    i, total = 0, 0
    new = []
    for x in range(step, h - step, step):
        for y in range(step, w - step, step):
            gradientX = np.sum(
                image[x-step:x+step+1, y-step:y+step+1] * sobelX)
            gradientY = np.sum(
                image[x-step:x+step+1, y-step:y+step+1] * sobelY)
            number = np.sqrt(gradientX ** 2 + gradientY ** 2)
            if number == 0:
                continue
            else:
                total += number
            if i % 1000 == 0:
                new.append(total/1000)
                total = 0
            i += 1
    return np.array(new)


def calcArea(image):
    area = 0
    hi, wi = image.shape
    for x in range(hi):
        for y in range(wi):
            area += image[x, y]
    return area


def normalize(lst, string):
    lst = np.array(lst)
    lst = lst.reshape(1, -1)
    lst = preprocessing.normalize(lst, norm="l2")
    lst = lst.reshape(-1)
    return lst


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1


gray_level = 8


def getGLCM(image, dx, dy):
    src = np.copy(image)
    ret = np.zeros((gray_level, gray_level))
    h, w = src.shape
    maxGrayLevel = np.amax(image) + 1

    src = src.astype("float32")
    if maxGrayLevel > gray_level:
        src = src * gray_level / maxGrayLevel
    src = src.astype("uint8")

    if dx == -1:
        for y in range(h - dy):
            for x in range(1, w):
                row = src[y, x]
                col = src[y + dy, x + dx]
                ret[row, col] += 1
    else:
        for y in range(h - dy):
            for x in range(w - dx):
                row = src[y, x]
                col = src[y + dy, x + dx]
                ret[row, col] += 1
    ret = ret / (h * w)
    return ret


def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i-j)*(i-j)*p[i][j]
            Asm += p[i][j]*p[i][j]
            Idm += p[i][j]/(1+(i-j)*(i-j))
            if p[i][j] > 0.0:
                Eng += p[i][j]*math.log(p[i][j])
    return [Asm, Con, Idm, -Eng]


# def torchConv2d(image, out_channel=1):
#     n = len(image.shape)
#     if n > 2:
#         h, w, c = image.shape
#         image = image.reshape((1, c, h, w))
#     else:
#         h, w = image.shape
#         c = 1
#         image = image.reshape((1, 1, h, w))
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     image = torch.tensor(image, device=device).float()
#     conv = torch.nn.Conv2d(in_channels=c, out_channels=out_channel, kernel_size=(3, 3))
#     maxP = torch.nn.MaxPool2d(kernel_size=2)
#     res = maxP(conv(image))
#     new = res.detach().numpy()
#     size, c, h, w = new.shape
#     if c > 2:
#         result = np.zeros((h, w, c))
#         for i in range(c):
#             result[:,:,i] = new[0,i]
#     else:
#         result = np.zeros((h, w))
#         result = new[0,0]
#     return result
