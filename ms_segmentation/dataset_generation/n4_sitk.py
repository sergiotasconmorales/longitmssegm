# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script to apply N4 bias field removal to images of the INSPIRATION dataset, which were previously generated using build_cross_sectional_dataset_proc_flair2D.py
#               and skull-stripped using HD-BET (apply_hd_bet_t1 and apply_brain_mask)
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      Script built based on SITK documentation https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
#
# --------------------------------------------------------------------------------------------------------------------

import os 
from os.path import join as jp
import SimpleITK as sitk
from general.general import list_files_with_extension, list_folders

path_data = r'D:\dev\CROSS_SECTIONAL3'
number_fitting_levels = 4
num_iterations = 100

#For each case, apply N4
cases_list = list_folders(path_data)
for case in cases_list[14:]:          
    print("-----------------------------------")      
    print("Current case: ", case)     
    print("-----------------------------------")  
    #Read images
    flair_img = sitk.ReadImage(jp(path_data, case, "FLAIR_masked.nii.gz"))
    t1c_img = sitk.ReadImage(jp(path_data, case, "T1_c_masked.nii.gz"))
    t1_img = sitk.ReadImage(jp(path_data, case, "T1_masked.nii.gz"))
    brain_mask = sitk.ReadImage(jp(path_data, case, "T1_bet_mask.nii.gz"))

    #Cast images to FLoat32 (required by library)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    flair_img_casted = caster.Execute(flair_img)
    t1c_img_casted = caster.Execute(t1c_img)
    t1_img_casted = caster.Execute(t1_img)
    caster_mask = sitk.CastImageFilter()
    caster_mask.SetOutputPixelType(sitk.sitkUInt8)
    brain_mask_casted = caster_mask.Execute(brain_mask)
    brain_mask_casted.SetOrigin((0,0,0))
    brain_mask_casted.SetSpacing((0.9375, 0.9375, 3))

    #Apply bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #corrector.SetMaximumNumberOfIterations(num_iterations*number_fitting_levels)
    print("Correcting FLAIR...")
    flair_img_corrected = corrector.Execute(flair_img_casted, brain_mask_casted)
    print("Correcting T1_c...")
    t1c_img_corrected = corrector.Execute(t1c_img_casted, brain_mask_casted)
    print("Correcting T1...")
    t1_img_corrected = corrector.Execute(t1_img_casted, brain_mask_casted)

    #Copy metadata and save
    flair_img_corrected.CopyInformation(flair_img)
    t1c_img_corrected.CopyInformation(t1c_img)
    t1_img_corrected.CopyInformation(t1_img)
    sitk.WriteImage(flair_img_corrected, jp(path_data, case, "FLAIR_masked_n4.nii.gz"))
    sitk.WriteImage(t1c_img_corrected, jp(path_data, case, "T1_c_masked_n4.nii.gz"))
    sitk.WriteImage(t1_img_corrected, jp(path_data, case, "T1_masked_n4.nii.gz"))

    
