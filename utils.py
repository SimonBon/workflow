import os
from typing import Union
import glob
import natsort
import cv2
import pandas as pd


class Sample():

    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.sample_name = os.path.split(sample_path)[-1]
        self._markers = ['Iridium_1033((1254))Ir193', 'Iridium_1033((1253))Ir191', 'H4K12Ac_2023((3829))Er167', 'Histone_1978((3831))Nd146']

        try:
            self.mcd_filepath = glob.glob(os.path.join(sample_path, '*.[mM][cC][dD]'))[0]
        except:
            raise Exception(f"No ROIs found in '{sample_path}'")

        try:
            roi_tmp = glob.glob(os.path.join(sample_path, '*[rR][oO][iI]*'))[0]
        except:
            raise Exception(f"No ROIs found in '{sample_path}'")

        roi_tmp = os.path.join(roi_tmp, os.listdir(roi_tmp)[0])
        roi_tmp = [os.path.join(roi_tmp, x) for x in os.listdir(roi_tmp) if not x.startswith(".")]

        roi_files = []
        for f in roi_tmp:
            imgs = natsort.natsorted([os.path.join(f, x) for x in os.listdir(f)])
            roi_files.append({
                "roi_num": f.split("/")[-1],
                "if_b": imgs[0],
                "if_g": imgs[1],
                "if_r": imgs[2]})

        rois = pd.DataFrame(roi_files)

        # aus IMC Daten die ROIs extrahieren und dann auch in
        # dataframe einspeichern und mergen und dann in diee ROI
        # class eingeben damit man auf die schöner zugreifen kann

        self.rois = self._get_rois()

    class ROI():
        def __init__(self, roi_num, b, g, r):
            self.roi_num = roi_num
            self.b = cv2.cvtColor(cv2.imread(b), cv2.COLOR_BGR2GRAY)
            self.g = cv2.cvtColor(cv2.imread(g), cv2.COLOR_BGR2GRAY)
            self.r = cv2.cvtColor(cv2.imread(r), cv2.COLOR_BGR2GRAY)

    def _get_rois(self):
        pass
