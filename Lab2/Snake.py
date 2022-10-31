# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:28:08 2022

@author: hugos
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io

plt.close('all')

# img = data.astronaut()
# img = rgb2gray(img)
img=io.imread('Images/Brain.tif')

s = np.linspace(0, 2*np.pi, 400)
r = 145 + 45*np.sin(s)
c = 175 + 45*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(img, 3, preserve_range=False),
                       init, alpha=0.0001, beta=10, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()