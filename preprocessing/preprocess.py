import numpy as np
from pybasic import correct_illumination, basic
from scipy.ndimage import median_filter
import random
import cv2
from time import time


def preprocess(img, percentage=1):

    #img = rm_hotpixel(img)
    #ff, bg = basic([img], verbosity=False)
    #img = correct_illumination([img], ff, bg)[0]
    img = ((img / img.max())*255).astype(np.uint8)
    b1 = np.percentile(img, percentage)
    t1 = np.percentile(img, 100-percentage)
    img = (img-b1)/(t1-b1)
    img = np.clip(img, 0, 1)

    return (img*255).astype(np.uint8)


def rm_hotpixel(img, threshold=0.9):

    blurred = median_filter(img, size=3)
    diff = img-blurred
    spots = np.array(np.where(diff > threshold*img.max())).T

    for spot in spots:
        img = handle_hotpixel(spot, img)

    print(f"{spots.shape[0]} hot pixels found.")
    return img


def handle_hotpixel(spot, img, sz=(1, 1)):

    ret = np.copy(img)
    ret[spot[0], spot[1]] = np.nan

    x0 = spot[1]-1 if spot[1] >= sz[1] else 0
    y0 = spot[0]-1 if spot[0] >= sz[0] else 0
    x1 = spot[1]+2 if spot[1] <= ret.shape[1]-sz[1] else ret.shape[1]
    y1 = spot[0]+2 if spot[0] <= ret.shape[0]-sz[1] else ret.shape[0]

    tmp = ret[y0:y1, x0:x1]
    mean = np.nanmean(tmp)
    ret[spot[0], spot[1]] = mean
    return ret
