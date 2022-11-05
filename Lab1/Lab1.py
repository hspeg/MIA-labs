# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:37:23 2022

@author: hugos
"""

from random import gauss, randint

##############################################################################

def gaussianNoise(image, sigma):
    r"""
    Parameters
    ----------
    image : ndarray
        Input image to be noised, which should be 2D and grayscale.
    sigma : float
        Standard deviation of gaussian distribution used for noising.

    Returns
    -------
    im : ndarray
        Noised image, of same shape as `image`.

    """
    im = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res = -1
            while res < 0 or res > 256 :
                res = gauss(image[i,j], sigma)
            im[i,j] = res 
    return im

def saltAndPepperNoise(image,rang):
    r"""
    Parameters
    ----------
    image : 2D ndarray
        Input image to be noised, which should be 2D and grayscale.
    rang : int
        Number of iteration used for noising.

    Returns
    -------
    im : TYPE
        Noised image, of same shape as `image`.

    """
    im = image.copy()
    for i in range(rang):
        color = 0
        if(randint(0, 1)): 
            color = 255
        i = randint(0, image.shape[0]-1)
        j = randint(0, image.shape[1]-1)
        im[i,j] = color
    return im
   
##############################################################################