import cv2 as cv
from cv2 import add
import numpy as np
from time import time
from workflow.utils import additive_blend
import matplotlib.pyplot as plt

class FeatureExtractor():

    def __init__(self, type, *args, **kwargs) -> None:
        self.type = type.upper()

        if self.type == "SIFT":
            self.extractor = cv.SIFT_create(*args, **kwargs)
            self.norm = cv.NORM_L2

        elif self.type == "ORB":
            self.extractor = cv.ORB_create(*args, **kwargs)
            self.norm = cv.NORM_HAMMING

        elif self.type == "AKAZE":
            self.extractor = cv.AKAZE_create(*args, **kwargs)
            self.norm = cv.NORM_HAMMING

        elif self.type == "BRISK":
            self.extractor = cv.BRISK_create(*args, **kwargs)
            self.norm = cv.NORM_HAMMING

        else:
            raise ValueError(f"{type} is not a valid feature detection algorithm. Chose one of 'SIFT, ORB, AKAZE, BRISK'")


    def __call__(self, im0, im1) -> None:
        self.im0 = im0
        self.im1 = im1

        self.kp0, self.des0 = self.extractor.detectAndCompute(im0, None)
        self.kp1, self.des1 = self.extractor.detectAndCompute(im1, None)


    def match(self, ratio_thresh=0.7) -> None:
        matcher = cv.BFMatcher(self.norm) 
        knn_matches = matcher.knnMatch(self.des0, self.des1, 2)
        self.matches = [[m] for m, n in knn_matches if m.distance < ratio_thresh*n.distance]
    

    def estimate(self) -> None:
        self.points = np.array([np.array((self.kp0[m[0].queryIdx].pt, self.kp1[m[0].trainIdx].pt)) for m in self.matches]).astype(np.int64)
        print(f"Found {len(self.points)} matches.")
        if self.points.size != 0:
            h, _ = cv.estimateAffinePartial2D(self.points[:, 0], self.points[:, 1])
            self.h = h
        else:
            self.h = None

    def draw_matches(self) -> None:
        matched_image = cv.drawMatchesKnn(self.im0, self.kp0, self.im1, self.kp1, self.matches, None, flags=2)
        plt.imshow(matched_image)
        plt.show() 

    def show_overlay(self):
        if hasattr(self, "fixed"):
            if isinstance(self.fixed, np.ndarray):
                plt.imshow(additive_blend(self.fixed, self.warped))
                plt.show()


    def warp(self) -> None:
        if isinstance(self.h, np.ndarray):
            self.fixed = self.im1
            self.warped = cv.warpAffine(self.im0, self.h, (self.fixed.shape[0], self.fixed.shape[1]))

        else:
            self.fixed, self.warped = None, None