import os
import random
import torch
import nibabel as nib
import numpy as np  
import pandas as pd
from ms_segmentation.general.general import save_image, get_groups, parse_log_file, create_folder, list_folders, save_image, get_experiment_name, create_log, cls, get_dictionary_with_paths, list_files_with_name_containing
from os.path import join as jp
from ms_segmentation.plot.plot import shim_slice, shim_overlay_slice, shim, shim_overlay, plot_learning_curve
from medpy.io import load
from ms_segmentation.data_generation.patch_manager_3d import normalize_data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adadelta, Adam
import torchvision.transforms as transforms
from ms_segmentation.general.training_helper import EarlyStopping, exp_lr_scheduler, dice_loss, create_training_validation_sets, get_dictionary_with_paths_cs
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score as acc
from ms_segmentation.evaluation.metrics import compute_metrics
from torch.optim import Adadelta, Adam
import cc3d

path_data = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\cross_sectional'
path_new_data = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\histogram_matched'

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)

patients = list_folders(path_data)

for pat in patients: # for each patient
    print("Patient:", pat)
    timepoints = list_folders(jp(path_data, pat))
    create_folder(jp(path_new_data, pat))
    #take first point as reference
    ref_flair = normalize_data(nib.load(jp(path_data, pat, timepoints[0], 'flair.nii.gz')).get_fdata())
    ref_mprage = normalize_data(nib.load(jp(path_data, pat, timepoints[0], 'mprage.nii.gz')).get_fdata())
    ref_pd = normalize_data(nib.load(jp(path_data, pat, timepoints[0], 'pd.nii.gz')).get_fdata())
    ref_t2 = normalize_data(nib.load(jp(path_data, pat, timepoints[0], 't2.nii.gz')).get_fdata())
    brain_mask_ref = nib.load(jp(path_data, pat, timepoints[0], 'brain_mask.nii.gz')).get_fdata()
    #mask1 = nib.load(jp(path_data, pat, timepoints[0], 'mask1.nii.gz')).get_fdata()
    #mask2 = nib.load(jp(path_data, pat, timepoints[0], 'mask2.nii.gz')).get_fdata()

    #Save first timepoint without modifying it
    create_folder(jp(path_new_data, pat, timepoints[0]))
    save_image(ref_flair, jp(path_new_data, pat, timepoints[0], "flair.nii.gz"))
    save_image(ref_mprage, jp(path_new_data, pat, timepoints[0], "mprage.nii.gz"))
    save_image(ref_pd, jp(path_new_data, pat, timepoints[0], "pd.nii.gz"))
    save_image(ref_t2, jp(path_new_data, pat, timepoints[0], "t2.nii.gz"))
    save_image(brain_mask_ref, jp(path_new_data, pat, timepoints[0], "brain_mask.nii.gz"))
    #save_image(mask1, jp(path_new_data, pat, timepoints[0], "mask1.nii.gz"))
    #save_image(mask2, jp(path_new_data, pat, timepoints[0], "mask2.nii.gz"))


    for tp in timepoints[1:]: # from second timepoint
        print("Current tp: ", tp)
        target_flair = normalize_data(nib.load(jp(path_data, pat, tp, 'flair.nii.gz')).get_fdata())
        target_mprage = normalize_data(nib.load(jp(path_data, pat, tp, 'mprage.nii.gz')).get_fdata())
        target_pd = normalize_data(nib.load(jp(path_data, pat, tp, 'pd.nii.gz')).get_fdata())
        target_t2 = normalize_data(nib.load(jp(path_data, pat, tp, 't2.nii.gz')).get_fdata())
        target_brain_mask= nib.load(jp(path_data, pat, tp, 'brain_mask.nii.gz')).get_fdata()

        matched_flair = _match_cumulative_cdf(target_flair, ref_flair)
        matched_mprage = _match_cumulative_cdf(target_mprage, ref_mprage)
        matched_pd = _match_cumulative_cdf(target_pd, ref_pd)
        matched_t2 = _match_cumulative_cdf(target_t2, ref_t2)
        create_folder(jp(path_new_data, pat, tp))
        save_image(matched_flair, jp(path_new_data, pat, tp, "flair.nii.gz"))
        save_image(matched_mprage, jp(path_new_data, pat, tp, "mprage.nii.gz"))
        save_image(matched_pd, jp(path_new_data, pat, tp, "pd.nii.gz"))
        save_image(matched_t2, jp(path_new_data, pat, tp, "t2.nii.gz"))
        save_image(target_brain_mask, jp(path_new_data, pat, tp, "brain_mask.nii.gz"))


