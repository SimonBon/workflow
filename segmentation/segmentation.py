import os
import pickle
from glob import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
import utils
from importlib import import_module


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("-r", "--roi", nargs="+")
    parser.add_argument("-o", "--option", required=True)
    parser.add_argument("--save_dir")
    args = parser.parse_args()
    return args


class Segmentation():

    def __init__(self, option, mpp=None, nuclear_diameter=10, segmentation_kwargs={}):

        if option.lower() == "mesmer":
            self.option = option
            self.__module = import_module("deepcell.applications")
            self.algorithm = self.__module.Mesmer(**segmentation_kwargs)
            print("Using Mesmer for segmentation!")

        elif option.lower() == "cellpose":
            self.option = option
            self.__module = import_module('cellpose.models')
            import torch
            self.algorithm = self.__module.Cellpose(gpu=torch.cuda.is_available(), model_type='nuclei', **segmentation_kwargs)
            print("Using Cellpose for segmentation!")

        else:
            raise ValueError(f"'{option}' is not valid! Only 'Mesmer' and 'Cellpsose' are valid entries for 'option'!")

        self.mpp = mpp
        self.nuclear_diameter = nuclear_diameter

    def __call__(self, image):

        img = np.expand_dims(np.stack((image, np.zeros_like(image)), -1), 0)

        if self.option.lower() == "mesmer":
            print("Predicting using Mesmer")
            return self.algorithm.predict(img, compartment='nuclear', image_mpp=self.mpp).squeeze()

        elif self.option.lower() == "cellpose":
            print("Predicting using Cellpose")
            diameter_px = self.nuclear_diameter / self.mpp
            channels = [[0, 0]]
            return self.algorithm.eval(img, diameter=diameter_px, channels=channels)[0].squeeze()
