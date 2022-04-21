import registration.register as reg
import importlib
importlib.reload(reg)
from time import time
import os
from Sample import Sample
import numpy as np
from preprocessing.preprocess import preprocess
from tqdm import tqdm

base = "/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/Multimodal_Imaging_Daria"
examples = [os.path.join(base, x) for x in os.listdir(base) if "BM" in x]

with open("/home/simon_g/workflow/results.txt", "w+") as fout:

    for data in tqdm(examples):
        try:
            S = Sample(data)
        except Exception as error:
            print(error)
            continue

        for R in S.rois:
            IF = np.rot90(preprocess(R.if_nuc))
            IMC = preprocess(R.imc_nuc_upscaled)

            for type, n in zip(["sift", "orb", "akaze"], [None, 1000000, None]):
                s = time()
                if n:
                    fex = reg.FeatureExtractor(type, nfeatures=n)
                else: 
                    fex = reg.FeatureExtractor(type)
                fex(IF,IMC)
                fout.writelines(f"{data};{R.roi_num};{type};{round(time()-s,2)},{len(fex.kp0)}\n")
