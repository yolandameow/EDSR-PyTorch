
# coding: utf-8

# In[1]:


from pylab import *
import numpy as np
import os
import shutil
import numpy as np

# generate the masked cityscape iamge

'''
[220,220,  0, 255]  # traffic sign
[  0,  0,142, 255]  # car
[  0,  0, 70, 255]  # truck
[  0, 60,100, 255]  # bus
[  0,  0, 90, 255]  # caravan
[  0,  0,110, 255]  # trailer
[  0, 80,100, 255]  # train
[  0,  0,230, 255]  # motorcycle
[119, 11, 32, 255]  # bicycle
[220, 20, 60, 255]  # people
'''
def image_mask(input_image_path,
               input_mask_path,
               masked_output_path,
               bw_output_path):
    counter = 0
    inputiamgefile = os.listdir(input_image_path)
    inputmaskfile = os.listdir(input_mask_path)
    for i in inputiamgefile:
        masktmp = "_".join(i.split(".")[0].split("_")[0:3] + ["gtFine_color"]) + ".png"
        im = imread(input_image_path + i)         #RGB
        im2 = im.copy()
        mask = imread(input_mask_path + masktmp)  # RGBA
        for j in range(im.shape[0]):
            for k in range(im.shape[1]):
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
                        ):                                          # which we left remain in the image
                    im[j,k,:] = [0,0,0]
                    im2[j,k,:] = [0,0,0]
                else:
                    im2[j,k,:] = [255,255,255]
        imsave(masked_output_path + "_".join(i.split(".")[0].split("_")[0:3] + ["masked"]) + ".png",im)
        imsave(bw_output_path + "_".join(i.split(".")[0].split("_")[0:3] + ["bw"]) + ".png",im2)
        counter += 1
        print(counter)

