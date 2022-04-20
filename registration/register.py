import cv2
import numpy as np


def find_matches(img0, img1, distance_match=0.8, max_features=30000, return_matches=False):

    sift = cv2.SIFT_create(nfeatures=max_features)

    kp1, des1 = sift.detectAndCompute(img0, None)
    kp2, des2 = sift.detectAndCompute(img1, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [[m] for (m, n) in matches if m.distance < distance_match*n.distance]
    points = np.array([np.array((kp1[m[0].queryIdx].pt, kp2[m[0].trainIdx].pt)) for m in good]).astype(np.int64)
    kp_match = cv2.drawMatchesKnn(img0, kp1, img1, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good) < 10:
        print(f"Found only {len(good)} matches.\n")

    else:

        print(f"Found {len(good)} matches.")
        h, _ = cv2.estimateAffinePartial2D(points[:, 0], points[:, 1])
        if return_matches:
            return kp_match, h
        else:
            return h


def transform(img0, img1, h):
    return cv2.warpAffine(img0, h, (img1.shape[1], img1.shape[0]))
