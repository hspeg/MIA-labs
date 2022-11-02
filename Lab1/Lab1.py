# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:37:23 2022

@author: hugos
"""

from skimage import io 
from skimage import restoration as res
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from random import gauss, randint
from medpy.filter.smoothing import anisotropic_diffusion


plt.close('all')

def gaussianNoise(image,sigma):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res = -1
            while res < 0 or res > 256 :
                res = gauss(image[i,j], sigma)
            image[i,j] = res 
    return image

def saltAndPepperNoise(image,rang):
    for i in range(rang):
        color = 0
        if(randint(0, 1)): 
            color = 255
        i = randint(0, image.shape[0]-1)
        j = randint(0, image.shape[1]-1)
        image[i,j] = color
    return image

def showHisto(image):
    plt.figure()
    plt.title("Histogram")
    plt.hist(image.ravel(), bins = 256)
    
def showImageGray(image,string):
    plt.figure()
    plt.title(string)
    plt.imshow(image, cmap='gray')
    
def performance(image1, image2):
    return 0
 
#im = data.coins()   
#im = io.imread('Images/Brain.tif')
im = io.imread('Images/Chest_CT.tif')

showImageGray(im,'Brain')

im_gauss = gaussianNoise(im, 50)
im_salt = saltAndPepperNoise(im, 100)

showImageGray(im_gauss, 'Gaussian noise')
showImageGray(im_salt, 'Salt and pepper noise')

##############################################################################

#im_nlm_filter = res.denoise_nl_means(im_gauss)
#im_nlm_filter = res.denoise_nl_means(im_salt_and_pepper)
#showImageGray(im_nlm_filter,'NLM filter')

##############################################################################

# im_aniso = anisotropic_diffusion(im)
# showImageGray(im_aniso,'Anisotropic filter')

