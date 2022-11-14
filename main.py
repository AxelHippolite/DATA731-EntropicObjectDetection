from object_detection import *
from registration import *
from image import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

def nearest(balls_centers, key_center):
    dist = [np.sqrt((i[0] - key_center[0])**2 + (i[1] - key_center[1])**2) for i in balls_centers]
    return min(range(len(dist)), key=dist.__getitem__)

if __name__ == '__main__':
    print('Code Started...')
    ratio = 0.25 #float(input('Enter a Ratio for Resizing : '))
    path_ref = 'dataset/data_o.jpg' #input('PATH Reference : ')
    path_src = 'dataset/data_w.jpg' #input('PATH Source : ')

    source, reference = resize(cv2.imread(path_src), ratio), resize(cv2.imread(path_ref), ratio)
    print('Size Test :', source.shape == reference.shape)

    print('Registration In Progress...')
    ref, src, center_key, contour_src = registration(reference, source)
    #cv2.circle(src, center_key, 5, (0, 255, 255), -1)
    print('Registration Finished...')

    print('Enthropy Detection In Progress...')
    res = enthropy_detection(ref, src)
    centers, contours = center_detection(res)
    print('Centers :', centers)

    print('Finding the Nearest...')
    nearest_point = nearest(centers, center_key)
    print('Displaying Resutls...')
    (x, y), radius = cv2.minEnclosingCircle(contour_src[0])
    cv2.circle(src, (int(x), int(y)), int(radius), (0, 255, 255), -1)
    for i in range(len(contours)):
        if i == nearest_point:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])
            cv2.circle(src, (int(x), int(y)), int(radius), (0, 255, 0), 3)
        else:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])
            cv2.circle(src, (int(x), int(y)), int(radius), (255, 0, 0), 3)
    print('Completed...')
    plt.imshow(src)
    plt.show()