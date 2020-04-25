import numpy as np


def lbpMasks(image):
    h, w = image.shape
    new = np.zeros((h+2,w+2), dtype="uint8")
    new[1:h+1,1:w+1] = np.copy(image)
    
    for y in range(1, h+1):
        for x in range(1, w+1):
            center = new[y,x]
            print(center)
            print( new[y-1:y+2,x-1:x+2] )
            
            break
        break
    return new