# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script to apply brain masks generated with HD-BET on T1 images, to all images (FLAIR, T1c and T1). All of this for INSPIRATION data which was previously generated
#               as a cross-sectional database using script build_cross_sectional_dataset_proc_flair2D.py. Mask applied also on T1 because output of HD-BET has modified intensities
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------

import os
import SimpleITK as sitk
from os.path import join as jp
from general.general import list_files_with_extension, list_folders

path_data = r'D:\dev\CROSS_SECTIONAL3' #Path to cross-sectional dataset generated with build_cross_sectional_dataset_proc_flair2D.py

cases_list = list_folders(path_data)
for case in cases_list:
    print("Current case: ", case, "/", len(cases_list))

    #Read mask obtained with HD-BET
    brain_mask = sitk.ReadImage(jp(path_data, case, "T1_bet_mask.nii.gz"))
    flair_img = sitk.ReadImage(jp(path_data, case, "FLAIR.nii.gz"))
    t1c_img = sitk.ReadImage(jp(path_data, case, "T1_c.nii.gz"))
    t1_img = sitk.ReadImage(jp(path_data, case, "T1.nii.gz"))
    pd_img = sitk.ReadImage(jp(path_data, case, "PD.nii.gz"))
    t2_img = sitk.ReadImage(jp(path_data, case, "T2.nii.gz"))


    #Cast images to integer 32 
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkInt32)
    flair_img_casted = caster.Execute(flair_img)
    t1c_img_casted = caster.Execute(t1c_img)
    t1_img_casted = caster.Execute(t1_img)
    pd_img_casted = caster.Execute(pd_img)
    t2_img_casted = caster.Execute(t2_img)


    #Apply masks
    mult_filter = sitk.MultiplyImageFilter()
    flair_masked = mult_filter.Execute(flair_img_casted, brain_mask)
    t1c_masked = mult_filter.Execute(t1c_img_casted, brain_mask)
    t1_masked = mult_filter.Execute(t1_img_casted, brain_mask)
    pd_masked = mult_filter.Execute(pd_img_casted, brain_mask)
    t2_masked = mult_filter.Execute(t2_img_casted, brain_mask)


    #Copy information
    flair_masked.CopyInformation(flair_img)
    t1c_masked.CopyInformation(t1c_img)
    t1_masked.CopyInformation(t1_img)
    pd_masked.CopyInformation(pd_img)
    t2_masked.CopyInformation(t2_img)


    #Save images
    sitk.WriteImage(flair_masked, jp(path_data, case, "FLAIR_masked.nii.gz"))
    sitk.WriteImage(t1c_masked, jp(path_data, case, "T1_c_masked.nii.gz"))
    sitk.WriteImage(t1_masked, jp(path_data, case, "T1_masked.nii.gz"))
    sitk.WriteImage(pd_masked, jp(path_data, case, "PD_masked.nii.gz"))
    sitk.WriteImage(t2_masked, jp(path_data, case, "T2_masked.nii.gz"))

