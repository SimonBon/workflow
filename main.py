import matplotlib.pyplot as plt
import preprocessing.preprocess as pp
import registration.register as reg
from Sample import Sample
from utils import additive_blend
import numpy as np

sample = "/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/Multimodal_Imaging_Daria/_tmp_simon/20211214_18-2600_BM"
rois = Sample(sample)

fig, ax = plt.subplots(5, len(rois.rois), figsize=(15, 10))

for i, roi in enumerate(rois.rois):

    pp_if = pp.preprocess(roi.if_nuc)
    pp_imc = pp.preprocess(roi.imc_nuc)

    ax[0, i].imshow(roi.if_nuc)
    ax[1, i].imshow(pp_if)
    ax[2, i].imshow(roi.imc_nuc)
    ax[3, i].imshow(pp_imc)

    h = reg.find_matches(pp_if, pp_imc)
    transformed = reg.transform(pp_if, pp_imc, h)

    ax[4, i].imshow(additive_blend(transformed, pp_imc))
    plt.tight_layout()

plt.show()
