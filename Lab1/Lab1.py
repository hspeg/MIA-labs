# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:37:23 2022

@author: hugos
"""

from skimage import io 
from skimage.restoration import denoise_nl_means
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from random import gauss, randint
from medpy.filter.smoothing import anisotropic_diffusion


plt.close('all')

def gaussianNoise(image,sigma):
    im = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res = -1
            while res < 0 or res > 256 :
                res = gauss(image[i,j], sigma)
            im[i,j] = res 
    return im

def saltAndPepperNoise(image,rang):
    im = image.copy()
    for i in range(rang):
        color = 0
        if(randint(0, 1)): 
            color = 255
        i = randint(0, image.shape[0]-1)
        j = randint(0, image.shape[1]-1)
        im[i,j] = color
    return im

def showHisto(image):
    plt.figure()
    plt.title("Histogram")
    plt.hist(image.ravel(), bins = 256)
    
def showImageGray(image,string):
    plt.figure()
    plt.title(string)
    plt.imshow(image, cmap='gray')
    
def performance(image1, image2): ####!!!!!!!###########
    return 0

def Question1(image):
    fig, axes = plt.subplots(2, 4, figsize=(8, 8), sharex=False, sharey=True)
    ax = axes.ravel()
    
    ax[0].set_title("Gauss, sigma = 10")
    ax[0].imshow(gaussianNoise(image, 10), cmap='gray')
    
    ax[1].set_title("Gauss, sigma = 20")
    ax[1].imshow(gaussianNoise(image, 20), cmap='gray')
    
    ax[2].set_title("Gauss, sigma = 40")
    ax[2].imshow(gaussianNoise(image, 40), cmap='gray')
    
    ax[3].set_title("Gauss, sigma = 80")
    ax[3].imshow(gaussianNoise(image, 80), cmap='gray')
    
    ax[4].set_title("Salt and pepper, range = 100")
    ax[4].imshow(saltAndPepperNoise(image, 100), cmap='gray')
    
    ax[5].set_title("Salt and pepper, range = 1000")
    ax[5].imshow(saltAndPepperNoise(image, 1000), cmap='gray')
    
    ax[6].set_title("Salt and pepper, range = 10000")
    ax[6].imshow(saltAndPepperNoise(image, 10000), cmap='gray')
    
    ax[7].set_title("Salt and pepper, range = 100000")
    ax[7].imshow(saltAndPepperNoise(image, 100000), cmap='gray')
    
    for a in ax:
        a.set_axis_off()
    
    plt.show()
    
def Question2(image):
    fig, axes = plt.subplots(3, 4, figsize=(8, 8), sharex=False, sharey=True)
    ax = axes.ravel()
    
    im1 = gaussianNoise(image, 10)
    im2 = gaussianNoise(image, 20)
    im3 = gaussianNoise(image, 40)
    im4 = gaussianNoise(image, 80)
    
    im5 = saltAndPepperNoise(image, 100)
    im6 = saltAndPepperNoise(image, 1000)
    im7 = saltAndPepperNoise(image, 10000)
    im8 = saltAndPepperNoise(image, 100000)
    
    ax[0].set_title("NLM filtering gauss")
    ax[0].imshow(denoise_nl_means(im1), cmap='gray')
    
    ax[1].imshow(denoise_nl_means(im2), cmap='gray')
    
    ax[2].imshow(denoise_nl_means(im3), cmap='gray')
    
    ax[3].imshow(denoise_nl_means(im4), cmap='gray')
    
    ax[4].set_title("NLM filtering S&P")
    ax[4].imshow(denoise_nl_means(im5), cmap='gray')
    
    ax[5].imshow(denoise_nl_means(im6), cmap='gray')
    
    ax[6].imshow(denoise_nl_means(im7), cmap='gray')
    
    ax[7].imshow(denoise_nl_means(im8), cmap='gray')
    
    ax[8].set_title("NLM filtering S&P")
    ax[8].imshow(denoise_nl_means(im5), cmap='gray')
    
    ax[9].imshow(denoise_nl_means(im6), cmap='gray')
    
    ax[10].imshow(denoise_nl_means(im7), cmap='gray')
    
    ax[11].imshow(denoise_nl_means(im8), cmap='gray')
    
    for a in ax:
        a.set_axis_off()
    
    plt.show()
    
def Question3(image):
    
    fig, axes = plt.subplots(2, 4, figsize=(8, 8), sharex=False, sharey=True)
    ax = axes.ravel()
    
    im1 = gaussianNoise(image, 10)
    im2 = gaussianNoise(image, 20)
    im3 = gaussianNoise(image, 40)
    im4 = gaussianNoise(image, 80)
    
    im5 = saltAndPepperNoise(image, 100)
    im6 = saltAndPepperNoise(image, 1000)
    im7 = saltAndPepperNoise(image, 10000)
    im8 = saltAndPepperNoise(image, 100000)
    
    ax[0].set_title("NLM filtering gauss")
    ax[0].imshow(anisotropic_diffusion(im1), cmap='gray')
    
    ax[1].imshow(anisotropic_diffusion(im2), cmap='gray')
    
    ax[2].imshow(anisotropic_diffusion(im3), cmap='gray')
    
    ax[3].imshow(anisotropic_diffusion(im4), cmap='gray')
    
    ax[4].set_title("NLM filtering S&P")
    ax[4].imshow(anisotropic_diffusion(im5), cmap='gray')
    
    ax[5].imshow(anisotropic_diffusion(im6), cmap='gray')
    
    ax[6].imshow(anisotropic_diffusion(im7), cmap='gray')
    
    ax[7].imshow(anisotropic_diffusion(im8), cmap='gray')
    
    for a in ax:
        a.set_axis_off()
    
    plt.show()
    
    
##############################################################################
 
im1 = io.imread('Images/Brain.tif')
im2 = io.imread('Images/Chest_CT.tif')

##############################################################################

#im11 = gaussianNoise(im1, 40)


im_aniso = anisotropic_diffusion(im1)

showImageGray(im_aniso,'Anisotropic filter')



################Question 1####################################################

#Question1(im1)

################Question 2####################################################

#Question2(im1)

################Question 3####################################################

#Question3(im1)
