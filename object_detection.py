from image import *
import numpy as np
import cv2

def opening(img, kernel, iter_e, iter_d):
    return cv2.dilate(cv2.erode(img, kernel, iterations=iter_e), kernel, iterations=iter_d)

def closing(img, kernel, iter_e, iter_d):
    return cv2.erode(cv2.dilate(img, kernel, iterations=iter_d), kernel, iterations=iter_e)

def pooling3(img, i, j):
    return img[i-1:i+1,j-1:j+1].flatten()

def dkl(var1, var2):
    return (1/2) * ((var2**2 / var1**2) + (var1**2 / var2**2))

def enthropy_detection(ref, src):
    res = np.zeros((src.shape[0], src.shape[1]))
    for i in range(1, src.shape[0]-1):
        for j in range(1, src.shape[1]-1):
            src_pool, ref_pool = pooling3(src, i, j), pooling3(ref, i, j)
            res[i, j] = np.log(dkl(np.var(src_pool), np.var(ref_pool)))
    return map(res)

def center_detection(img):
    centers = []
    res = np.where(img < np.amax(img)/2, 0, np.amax(img))
    cl = closing(res, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 15, 15)
    op = opening(cl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), 2, 2)
    contours, hierarchy = cv2.findContours(op.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
    return centers, contours
