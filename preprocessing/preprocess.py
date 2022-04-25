import numpy as np
from pybasic import correct_illumination, basic
import random
import cv2
from time import time


def preprocess(img, percentage=1):

    img = rm_hotpixel(img)
    ff, bg = basic([img], verbosity=False)
    img = correct_illumination([img], ff, bg)[0]
    img = ((img / img.max())*255).astype(np.uint8)
    b1 = np.percentile(img, percentage)
    t1 = np.percentile(img, 100-percentage)
    img = (img-b1)/(t1-b1)
    img = np.clip(img, 0, 1)

    return (img*255).astype(np.uint8)


def rm_hotpixel(img: np.ndarray, threshold=0.5) -> np.ndarray:
    
    #change datatype to float32 to be able to compute median filter
    ret = (img/img.max()*255).astype("float32")
    
    #calculate blurred version and substract it from original image
    blurred = cv2.medianBlur(ret,3)
    diff = ret-blurred

    #get spot idx where diff value > threshold
    spots = np.array(np.where(diff > threshold*ret.max())).T

    #remove each hotpixel defined in spot
    for spot in spots:
        ret[spot[0], spot[1]] = handle_hotpixel(spot, ret)

    #verbose 
    print(f"{spots.shape[0]} hot pixels found.")

    return ret


def handle_hotpixel(spot: np.ndarray, img: np.ndarray, sz=(1, 1)) -> np.float32:

    #get temporary image and set spot to NaN
    tmp_img = np.copy(img)
    tmp_img[spot[0], spot[1]] = np.nan

    #check if spot is on the boarder of the image and define the range where the mean is then calculated
    x0 = spot[1]-1 if spot[1] >= sz[1] else 0
    y0 = spot[0]-1 if spot[0] >= sz[0] else 0
    x1 = spot[1]+2 if spot[1] <= tmp_img.shape[1]-sz[1] else tmp_img.shape[1]
    y1 = spot[0]+2 if spot[0] <= tmp_img.shape[0]-sz[1] else tmp_img.shape[0]

    #calculate mean of surrounding area and return it
    tmp = tmp_img[y0:y1, x0:x1]
    mean = np.nanmean(tmp).astype(np.float32)

    return mean
