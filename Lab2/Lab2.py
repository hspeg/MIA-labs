# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:51:13 2022

@author: hugos
"""

import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
from skimage import io 
import numpy as np

plt.close('all')

im1 = io.imread('Images/Brain.tif')
im2 = io.imread('Images/Chest_CT.tif')

#Threshold#####################################################################

def showThreshold(image):

    thresh = threshold_otsu(image)
    binary = image > thresh
    
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')
    
    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')
    
    plt.show()
    
# showThreshold(im1)
# showThreshold(im2)

# from skimage.filters import try_all_threshold

# # img = io.imread('Images/Brain.tif')

# # fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
# # plt.show()

# #Region growing################################################################

from skimage.segmentation import flood_fill

def regionGrowing(image,x,y):
    light_segment = flood_fill(image, (y, x), 0, tolerance=10)
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    ax[1].imshow(light_segment, cmap=plt.cm.gray)
    ax[1].plot(x, y, 'ro')  # seed point
    ax[1].set_title('After flood fill')
    ax[1].axis('off')
    
    plt.show()

#regionGrowing(im2,451,426)

# #Watershed###################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def water(image):
    # Generate an initial image with two overlapping circles
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
    plt.show()
    
#water(im2)

#####Snake####################################################################

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io


def snake(image):
    s = np.linspace(0, 2*np.pi, 400)
    r = 380 + 20*np.sin(s)
    c = 493 + 20*np.cos(s)
    init = np.array([r, c]).T
    
    snake = active_contour(gaussian(image, 3, preserve_range=False),
                           init, alpha=0.0001, beta=10, gamma=0.001)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, image.shape[1], image.shape[0], 0])
    
    plt.show()
    
snake(im2)