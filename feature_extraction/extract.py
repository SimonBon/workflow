import queue
import threading
import sys
import utils
import matplotlib.pyplot as plt
import cv2
import pickle
import argparse
import numpy as np
import skimage.measure
import multiprocessing
import os
from glob import glob

NUM_WORKER = 16


def extract(segmentation_mask, marker_images, marker_list):

    num_cells = segmentation_mask.max()
    all_features = []
    cell_list = list(range(1, num_cells))
    ExtractionQueue = queue.Queue()
    for cell in cell_list:
        ExtractionQueue.put_nowait(cell)

    for _ in range(NUM_WORKER):
        Worker(
            ExtractionQueue,
            all_features,
            segmentation_mask,
            marker_images,
            marker_list,
            extract_features,
            cell_list,
            num_cells).start()

    ExtractionQueue.join()

    return all_features


class Worker(threading.Thread):
    def __init__(self, q, ret, mask, imgs, marker, task, list, n, *args, **kwargs):
        self.q = q
        self.ret = ret
        self.mask = mask
        self.imgs = imgs
        self.marker = marker
        self.task = task
        self.list = list
        self.n = n
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                cell = self.q.get_nowait()
                self.list.pop()
                self.__progressBar()
                self.ret.append(self.task(cell, self.mask, self.imgs, self.marker))

            except queue.Empty:
                return

            self.q.task_done()

    def __progressBar(self, barLength=50):

        percent = int((self.n - len(self.list)) * 100 / self.n)
        arrow = '|' * int(percent/100 * barLength - 1) + '|'
        spaces = ' ' * (barLength - len(arrow))

        sys.stdout.write("\r" + f'Progress: |{arrow}{spaces}| {percent}% [{self.n-len(self.list)}/{self.n}]' + "\r")
        sys.stdout.flush()


def get_expression_features(expression, features, cell_dict, marker):

    for feature in features:
        method = getattr(type(expression), feature)
        feature_val = method(expression)
        cell_dict[f"{marker}_{feature}"] = float(feature_val)


def get_morphological_features(cell_mask, features, cell_dict):
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops.py#L1001-L1294
    nzy, nzx = np.nonzero(cell_mask)
    cell_mask_crop = cell_mask[nzy.min():nzy.max()+1, nzx.min():nzx.max()+1]
    area = cell_mask_crop.sum()

    rprops = skimage.measure.regionprops(cell_mask_crop.astype(np.uint8))[0]
    for feature in features:
        feature_val = getattr(rprops, feature)
        try:
            if len(feature_val) >= 2:
                for i in range(len(feature_val)):
                    cell_dict[f"{feature}_{i}"] = feature_val[i]
            else:
                cell_dict[f"{feature}"] = feature_val
        except:
            cell_dict[f"{feature}"] = feature_val


def extract_features(cell, mask, imgs, markers):

    cell_mask = mask == cell

    exp_features = ["mean"]  # , "min", "max", "std"]
    morph_features = ["area", "major_axis_length", "minor_axis_length", "orientation", "perimeter"]
    #"centroid", "convex_area", "eccentricity", "equivalent_diameter", 'feret_diameter_max',

    cell_dict = {"cell": cell}

    for m in range(len(markers)):
        marker_exp = imgs[m][cell_mask]
        get_expression_features(marker_exp, exp_features, cell_dict, markers[m])

    get_morphological_features(cell_mask, morph_features, cell_dict)

    return cell_dict
