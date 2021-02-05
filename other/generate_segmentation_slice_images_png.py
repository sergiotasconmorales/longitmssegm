import os
from os.path import join as jp
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cc3d
from ms_segmentation.general.general import list_folders, list_files_with_name_containing, list_files_with_extension
from ms_segmentation.evaluation.metrics import compute_metrics

patient = "01"
slice_index = 88
alpha_ = 0.8



path_cs = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation'
path_l = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\cross_validation'
path_gt = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\isbi_train'
path_results = r'D:\dev\ms_data\Challenges\ISBI2015\slice-examples'
path_img = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\isbi_train'

exp_cs = ['CROSS_VALIDATION_UNet3D_2020-06-04_14_54_25[all_modalities_new]', 'CROSS_VALIDATION_UNet3D_2020-06-25_07_07_15[chi-square_norm_train]']
exp_l = ['CROSS_VALIDATION_UNetConvLSTM3D_2020-06-13_08_43_21[all_modalities_no_hist_matching_new]', 'CROSS_VALIDATION_UNetConvLSTM3D_2020-06-23_21_31_39[longitudinal_chisquare_normalization_new]']


def shim_overlay_slice(img, mask, slice_index, alpha = 0.7):
    masked = np.ma.masked_where(mask[:,:,slice_index] ==0, mask[:,:,slice_index])
    fig, ax = plt.subplots()
    ax.imshow(img[:,:, slice_index], 'gray', interpolation='none')
    ax.imshow(masked, 'hsv', interpolation='none', alpha=alpha)
    fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    plt.show()

def remove_small(mask, min_vol = 3):
    labels_out = cc3d.connected_components(mask.astype(np.uint8))
    for i_cc in np.unique(labels_out):
        if len(labels_out[labels_out == i_cc]) < min_vol:
            mask[labels_out == i_cc] = 0
    return mask


#First, generate images for cross-sectional
for exp in exp_cs:
    images = os.listdir(jp(path_cs, exp, 'fold' + patient, 'results', patient))
    flair = list_files_with_name_containing(jp(path_img, patient), "flair", "nii.gz")
    for i_img, (img, fl) in enumerate(zip(images, flair)):
        mask = nib.load(jp(path_cs, exp, 'fold' + patient, 'results', patient, img)).get_fdata()
        mask = remove_small(mask)
        flair_img = nib.load(fl).get_fdata()
        shim_overlay_slice(flair_img, mask, slice_index, alpha = alpha_)
        #plt.imsave(jp(path_results, exp + "_" + patient + "_slice" + str(slice_index+1) + "_" + str(i_img+1).zfill(2) + ".png"), mask[:,:,slice_index], cmap = "gray")


# Generate images for longitudinal 
for exp in exp_l:
    images = os.listdir(jp(path_l, exp, 'fold' + patient, 'results', patient))
    flair = list_files_with_name_containing(jp(path_img, patient), "flair", "nii.gz")
    for i_img, (img, fl) in enumerate(zip(images, flair)):
        mask = nib.load(jp(path_l, exp, 'fold' + patient, 'results', patient, img)).get_fdata()
        mask = remove_small(mask)
        flair_img = nib.load(fl).get_fdata()
        shim_overlay_slice(flair_img, mask, slice_index, alpha = alpha_)
        #plt.imsave(jp(path_results, exp + "_" + patient + "_slice" + str(slice_index+1) + "_" + str(i_img+1).zfill(2) + ".png"), mask[:,:,slice_index], cmap = "gray")


# Generate images for ground truth
images = list_files_with_name_containing( jp(path_gt, patient) , "mask1", "nii.gz")
flair = list_files_with_name_containing(jp(path_img, patient), "flair", "nii.gz")
for i_img, (img, fl) in enumerate(zip(images, flair)):
    mask = nib.load(img).get_fdata()
    mask = remove_small(mask)
    flair_img = nib.load(fl).get_fdata()
    shim_overlay_slice(flair_img, mask, slice_index, alpha = alpha_)
    #plt.imsave(jp(path_results, "gt_" + patient + "_slice" + str(slice_index+1)+ "_" + str(i_img+1).zfill(2) +  ".png"), mask[:,:,slice_index], cmap = "gray")

# MRI images
images = list_files_with_name_containing(jp(path_img, patient), "flair", "nii.gz")
for i_img, img in enumerate(images):
    volume = nib.load(img).get_fdata()
    #plt.imsave(jp(path_results, "flair_" + patient + "_slice" + str(slice_index+1)+ "_" + str(i_img+1).zfill(2) +  ".png"), volume[:,:,slice_index], cmap = "gray")