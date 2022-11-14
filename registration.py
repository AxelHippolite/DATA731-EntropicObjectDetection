import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialization(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]

def opening(img, kernel, iter_e, iter_d):
    return cv2.dilate(cv2.erode(img, kernel, iterations=iter_e), kernel, iterations=iter_d)

def get_keyPoint(r, g, b):
    mean_diff_r, mean_diff_g, mean_diff_b = r - np.mean(r), g - np.mean(g), b - np.mean(b)
    diffs = [mean_diff_r - mean_diff_g - mean_diff_b, 
            mean_diff_g - mean_diff_b - mean_diff_r, 
            mean_diff_b - mean_diff_r - mean_diff_g]
    min_layers = [np.amin(diffs[0]), np.amin(diffs[1]), np.amin(diffs[2])]
    index_min = min(range(len(min_layers)), key=min_layers.__getitem__)
    color_heatmap = np.where(diffs[index_min] < np.amin(diffs[index_min]) * 0.60, abs(diffs[index_min]), 0)

    cleaned_heatmap = opening(color_heatmap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1, 1)
    contours, hierarchy = cv2.findContours(cleaned_heatmap.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contours[0])
    if M['m00'] != 0:
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00'])), contours
    else:
        return None

def cut(reference, source, center_ref, center_src):
    diff_center = (center_ref[0] - center_src[0], center_ref[1] - center_src[1])
    if diff_center[0] > 0:
        for i in range(diff_center[0]):
            reference, source = np.delete(reference, 0, 0), np.delete(source, -1, 0)
    if diff_center[0] < 0:
        for i in range(abs(diff_center[0])):
            reference, source = np.delete(reference, -1, 0), np.delete(source, 0, 0)
    if diff_center[1] > 0:
        for i in range(diff_center[1]):
            reference, source = np.delete(reference, 0, 1), np.delete(source, -1, 1)
    if diff_center[1] < 0:
        for i in range(abs(diff_center[1])):
            reference, source = np.delete(reference, -1, 1), np.delete(source, 0, 1)
    return reference, source

def registration(reference, source):
    r_ref, g_ref, b_ref = initialization(reference)
    r_src, g_src, b_src = initialization(source)
    center_ref, contour_ref = get_keyPoint(r_ref, g_ref, b_ref)
    center_src, contour_src = get_keyPoint(r_src, g_src, b_src)
    print('Centers Before Registration :', center_ref, center_src)

    ref, src = cut(reference, source, center_ref, center_src)
    print('Size Test After Registration :', ref.shape == src.shape)

    return ref, src, center_src, contour_src