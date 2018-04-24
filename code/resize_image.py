#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 22:37:47 2018

@author: Heqi
"""

import imageio
import os
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import sys

data_folder = "original_image"
input_folder="input_image"

filenames = os.listdir(data_folder)
image_shape = [2048,1024]
scale = 4

for filename in filenames:
    image = imageio.imread(data_folder+'/'+filename)
    image_input = image.resize(int(image_shape[0]/scale), int(image_shape[1]/scale))
    imageio.imsave(input_folder+'/'+filename, image_input)
    
    