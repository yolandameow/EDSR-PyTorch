
# coding: utf-8

# In[38]:


from pylab import *
import pylab
import numpy as np
import os
import shutil
import random
import math
import numpy as np
import model_run
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d

# generate the masked cityscape iamge

# put all process together
def all_in_one(input_image_path,     
               model_input_path,
               input_mask_path,
               simplify_mask_path,
               fixed_im_output_path,
               bw_im_output_path,
               model_output_path,
               bicubic_output_path,
               r = 3):
    counter = 0
    input_image_file = os.listdir(input_image_path)
    for i in input_image_file:
        im = imread(input_image_path + i)                         # ground true image
        bw_im = np.empty_like(im)                                 # bw image
        HR_target_im = im.copy()                                  # target part of image in HR
        simplify_mask = np.empty_like(im, dtype = np.bool_)       # use simpilify_mask to reduce computation
        im_high, im_wide = im.shape[0], im.shape[1]
        masktmp = "_".join(i.split(".")[0].split("_")[0:3] + ["gtFine_color"]) + ".png"
        mask = imread(input_mask_path + masktmp)                  # input mask image
        
        # compute bicubic for none target image
        low_im_high, low_im_wide = math.floor(im_high/r), math.floor(im_wide/r)
        bicubic_im_high, bicubic_im_wide = low_im_high * r, low_im_wide * r
        xxx = np.arange(0, bicubic_im_wide + 1, r)
        yyy = np.arange(0, bicubic_im_high + 1, r)
        xnew = np.arange(0,bicubic_im_wide,1)
        ynew = np.arange(0,bicubic_im_high,1)
        bicubic_im = np.array(()).reshape(bicubic_im_high, bicubic_im_wide,0)
        for j in range(3):   # each channel
            tmp_channel = im[:,:,j]
            tmp_channel = gaussian_filter(tmp_channel,0.5)                                   # gaussian blur
            tmp_channel = imresize(tmp_channel,(low_im_high + 1, low_im_wide + 1))           # down_scale
            func = interp2d(xxx,yyy,tmp_channel, kind = "cubic")                             # cubic interpolation per channel
            tmp_channel = func(xnew, ynew).reshape(bicubic_im_high, bicubic_im_wide, 1)
            bicubic_im = np.concatenate((bicubic_im,tmp_channel),axis = 2)
            
        for j in range(im_high):
            for k in range(im_wide):
                if not(
                       np.all(mask[j,k,:] == [220,220,  0, 255]) \
                    or np.all(mask[j,k,:] == [  0,  0,142, 255]) \
                    or np.all(mask[j,k,:] == [  0,  0, 70, 255]) \
                    or np.all(mask[j,k,:] == [  0, 60,100, 255]) \
                    or np.all(mask[j,k,:] == [  0,  0, 90, 255]) \
                    or np.all(mask[j,k,:] == [  0,  0,110, 255]) \
                    or np.all(mask[j,k,:] == [  0, 80,100, 255]) \
                    or np.all(mask[j,k,:] == [  0,  0,230, 255]) \
                    or np.all(mask[j,k,:] == [119, 11, 32, 255]) \
                    or np.all(mask[j,k,:] == [220, 20, 60, 255])
                        ):                                                  # which we left remain in the image
                    HR_target_im[j,k,:] = [0,0,0]
                    bw_im[j,k,:] = [0,0,0]
                    simplify_mask[j,k,:] = [False,False,False]
                else:
                    simplify_mask[j,k,:] = [True,True,True]
                    bw_im[j,k,:] = [255,255,255]
                    
        LR_target_im = imresize(HR_target_im,(low_im_high, low_im_wide))    # target part of image in LR
        imsave(model_input_path + "_".join(i.split(".")[0].split("_")[0:3] + ["model_input"]) + ".png", LR_target_im)
        imsave(bw_im_output_path + "_".join(i.split(".")[0].split("_")[0:3] + ["bw"]) + ".png", bw_im)
        imsave(bicubic_output_path + "_".join(i.split(".")[0].split("_")[0:3] + ["bicubic"]) + ".png", bicubic_im)
        np.save(simplify_mask_path + "_".join(i.split(".")[0].split("_")[0:3] + ["simplify_mask"]) + ".npy",simplify_mask)
        counter += 1
        print("Image Preprocessing, No.%d completed!" % counter)
        
    '''
    # put image into model
    print("Model Processing!")
    model(model_input_path, model_output_path, r)
    print("Model Processing Compelete!")
    '''
    model_run.model(model_input_path,r)
    
    # compute fixed image
    model_output_file = os.listdir(model_output_path)
    for i in model_output_file:
        HR_output_im = imread(model_output_path + i)
        bicubic_im_tmp = "_".join(i.split(".")[0].split("_")[0:3] + ["bicubic"]) + ".png"
        bicubic_im = imread(bicubic_output_path + bicubic_im_tmp)
        simplify_mask = np.load(simplify_mask_path + "_".join(i.split(".")[0].split("_")[0:3] + ["simplify_mask"]) + ".npy")
        fixed_im = np.multiply(HR_output_im, simplify_mask) + np.multiply(bicubic_im, ~simplify_mask)
        imsave(fixed_im_output_path + "_".join(i.split(".")[0].split("_")[0:3] + ["fixed"]) + ".png", fixed_im)
    print("fixed image generated!")


# In[39]:

'''
all_in_one("G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\original image\\",     
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\input\\",
           "G:\\CSC2515_final\\cityscape dataset\\val_mask\\frankfurt\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\simplify\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\fixed image\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\bw\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\output\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\bicubic\\",
           r = 2)
'''
all_in_one("G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\original image\\",     
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\input\\",
           "G:\\CSC2515_final\\cityscape dataset\\val_mask\\frankfurt\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\simplify\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\fixed image\\",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\bw\\",
           "G:\\CSC2515_final\\EDSR-PyTorch-master\\experiment\\test\\results",
           "G:\\CSC2515_final\\cityscape dataset\\data_EDSR(1)\\bicubic\\",
           r = 2)


