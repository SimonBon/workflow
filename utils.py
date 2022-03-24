import numpy as np
import cv2
from skimage.segmentation import find_boundaries

def additive_blend(im0, im1):

    im0 = np.array(im0)
    im0 = im0/np.percentile(im0, 99)
    im0 = np.clip(im0, 0, 1)

    im1 = np.array(im1)
    im1 = im1/np.percentile(im1, 99)
    im1 = np.clip(im1, 0, 1)

    rgb = np.zeros((*im0.shape, 3))
    rgb[:, :, 0] = im0
    rgb[:, :, 1] = im1

    return rgb


def _rmlead(inp, char='0'):
    for idx, letter in enumerate(inp):
        if letter != char:
            return inp[idx:]


def plot_segmentation(nuc, segmentation):

    rgb_data = cv2.cvtColor(nuc, cv2.COLOR_GRAY2RGB)
    boundaries = np.zeros_like(nuc)
    overlay_data = np.copy(rgb_data)

    boundary = find_boundaries(segmentation, connectivity=0, mode='outer')
    boundaries[boundary > 0] = 1
    boundaries = cv2.dilate(boundaries, np.ones((2,2)))

    overlay_data[boundaries > 0] = (255,0,0)
    return overlay_data