import numpy as np
import cv2

def resize(img, ratio):
    return cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))

def map(img):
    max_array = np.max(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = int((img[i, j] * 255)/max_array)
    return img