import os
import cv2 as cv 
import numpy as np
import nibabel as nib
from os.path import join as jp
from ms_segmentation.general.general import save_image, list_folders


path_base = r'D:\dev\s.tasconmorales\feature_maps'
experiments = list_folders(path_base)
batch_size = 32
patch_slice = 16

all_fm = {}
for exp in experiments:
    all_fm[exp] = []
    for i in range(batch_size):
        all_fm[exp].append(nib.load(jp(path_base, exp, "inc_batch000_fm" + str(i).zfill(3) + ".nii.gz")).get_fdata()[:,:,patch_slice])

for k,v in all_fm.items():
    all_fm[k] = np.concatenate(v, axis = 1)

tot = np.concatenate(list(all_fm.values()), axis = 0)

def bring_to_256_levels(input_im):
    return (255*(input_im - np.min(input_im))/(np.max(input_im) - np.min(input_im))).astype(np.uint8)

toti = bring_to_256_levels(tot)

np.max(toti)

cv.imwrite("tot.png", toti)

a = 52354