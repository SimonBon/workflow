import numpy as np

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
