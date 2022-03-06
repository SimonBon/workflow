import numpy as np
from pybasic import correct_illumination, basic
from scipy.ndimage import median_filter


def preprocess(img):

    img = rm_hotpixel(img)
    ff, bg = basic([img], verbosity=False)
    img = correct_illumination([img], ff, bg)[0]
    b1 = np.percentile(img, 5)
    t1 = np.percentile(img, 95)
    img = (img-b1)/(t1-b1)
    img = np.clip(img, 0, 1)
    return (img*255).astype(np.uint8)


def rm_hotpixel(img, threshold=0.9):

    blurred = median_filter(img, size=3)
    diff = img-blurred
    spots = np.where(diff > threshold*img.max())
    print(f"{len(spots[0])} hot pixels found.")
    return img
