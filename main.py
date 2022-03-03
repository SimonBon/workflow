from registration.register import register
from utils import Sample
import matplotlib.pyplot as plt

sample = "/Users/simongutwein/ccriod/OneDrive - CCRI/Github/Data/20211214_07-4158_TU"
rois = Sample(sample)

plt.imshow(rois.rois[0].if_imgs[rois.rois[0].if_marker.index("CD56")])
plt.show()
