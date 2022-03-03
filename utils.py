import os
import re
from typing import Union
import glob
import natsort
import cv2
import pandas as pd
from readimc import MCDFile
import numpy as np


class Sample():

    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.sample_name = os.path.split(sample_path)[-1]

        self.__imc_markers = ['Iridium_1033((1254))Ir193', 'Iridium_1033((1253))Ir191', 'H4K12Ac_2023((3829))Er167', 'Histone_1978((3831))Nd146']
        self.__if_markers = ["DAPI", "GD2", "CD56"]  # -> hier mal nochmal Daria fragen

        try:
            self.mcd_filepath = glob.glob(os.path.join(sample_path, '*.[mM][cC][dD]'))[0]
        except:
            raise Exception(f"No ROIs found in '{sample_path}'")

        rois_if = self._get_if_rois()
        rois_imc = self._get_imc_rois()
        rois = pd.merge(rois_if, rois_imc, on="roi_num")

        self.rois = [self.ROI(x, self.__if_markers) for _, x in rois.iterrows()]

    def _get_if_rois(self):

        try:
            roi_tmp = glob.glob(os.path.join(self.sample_path, '*[rR][oO][iI]*'))[0]
        except:
            raise Exception(f"No ROIs found in '{self.sample_path}'")

        roi_tmp = os.path.join(roi_tmp, os.listdir(roi_tmp)[0])
        roi_tmp = [os.path.join(roi_tmp, x) for x in os.listdir(roi_tmp) if not x.startswith(".")]

        roi_files = []
        for f in roi_tmp:
            imgs = natsort.natsorted([os.path.join(f, x) for x in os.listdir(f)])
            roi_files.append({
                "roi_num": int(f.split("/")[-1]),
                "if_b": imgs[0],
                "if_g": imgs[1],
                "if_r": imgs[2]})

        return pd.DataFrame(roi_files)

    def _get_imc_rois(self):

        roi_images = []
        with MCDFile(self.mcd_filepath) as f:

            for acq in f.slides[0].acquisitions:
                try:
                    img = f.read_acquisition(acq)
                except:
                    continue
                roi_num = int(_rmlead(re.search('[0-9]+', acq.description).group()))
                marker = acq.channel_labels

                idxs = []
                for m in self.__imc_markers:
                    idxs.append(marker.index(m))

                tmp_img = np.zeros_like(img[0])
                for idx in idxs:
                    tmp_img += img[idx]

                mean_img = tmp_img/len(self.__imc_markers)
                roi_images.append({
                    "roi_num": roi_num,
                    "imc_img": mean_img,
                    "imc_marker": marker,
                    "imc_imgs": img
                })

            return pd.DataFrame(roi_images)

    class ROI():
        def __init__(self, df, if_marker):
            self.roi_num = df["roi_num"]
            self.if_nuc = cv2.cvtColor(cv2.imread(df["if_b"]), cv2.COLOR_BGR2GRAY)
            self.if_marker = if_marker
            self.if_imgs = np.array([cv2.cvtColor(cv2.imread(df[x]), cv2.COLOR_BGR2GRAY) for x in ["if_b", "if_g", "if_r"]])
            self.imc_nuc = df["imc_img"]
            self.imc_marker = df["imc_marker"]
            self.imc_imgs = df["imc_imgs"]


def _rmlead(inp, char='0'):
    for idx, letter in enumerate(inp):
        if letter != char:
            return inp[idx:]