# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script to apply HD-BET to T1 images for INSPIRATION data which was previously generated as a cross-sectional database 
#               using script build_cross_sectional_dataset_proc_flair2D.py. HD-BET must be previously installed 
#               
# Author:       Sergio Tascon Morales (Research inter at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------

import os 
from general.general import list_files_with_extension, list_folders, print_line
import SimpleITK as sitk
from os.path import join as jp


#Path to data
path_data = r'D:\dev\CROSS_SECTIONAL3'

#Go to path where hd-bet python script is located
os.chdir(r'C:\Users\s.morales\virtual_environments\ms_preprocessing\Scripts')

#For each case, apply HD-BET (around 40 secs of processing per case)
cases_list = list_folders(path_data)
for case in cases_list[200:]:                     
    path_t1 = os.path.join(path_data, case, 'T1.nii.gz')
    os.system('cmd /c "python hd-bet.py -i '+ path_t1 + '"')



#Apply brain masks
print_line()
print("Aplying brain masks to all images...")
print_line()
cases_list = list_folders(path_data)
for case in cases_list:                                       # ACHTUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUNG
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


#Apply N4

print_line()
print("Aplying N4 to all images...")
print_line()

number_fitting_levels = 4
num_iterations = 100

def correct_origin_and_spacing(the_image):
    the_image.SetOrigin((0,0,0))
    the_image.SetSpacing((0.9375, 0.9375, 3))

#For each case, apply N4
cases_list = list_folders(path_data)
for case in cases_list:          
    print("Current case: ", case, "/", len(cases_list))   
    #Read images
    flair_img = sitk.ReadImage(jp(path_data, case, "FLAIR_masked.nii.gz"))
    t1c_img = sitk.ReadImage(jp(path_data, case, "T1_c_masked.nii.gz"))
    t1_img = sitk.ReadImage(jp(path_data, case, "T1_masked.nii.gz"))
    pd_img = sitk.ReadImage(jp(path_data, case, "PD_masked.nii.gz"))
    t2_img = sitk.ReadImage(jp(path_data, case, "T2_masked.nii.gz"))
    brain_mask = sitk.ReadImage(jp(path_data, case, "T1_bet_mask.nii.gz"))

    #Cast images to FLoat32 (required by library)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    flair_img_casted = caster.Execute(flair_img)
    t1c_img_casted = caster.Execute(t1c_img)
    t1_img_casted = caster.Execute(t1_img)
    pd_img_casted = caster.Execute(pd_img)
    t2_img_casted = caster.Execute(t2_img)
    caster_mask = sitk.CastImageFilter()
    caster_mask.SetOutputPixelType(sitk.sitkUInt8)
    brain_mask_casted = caster_mask.Execute(brain_mask)
    brain_mask_casted.SetOrigin((0,0,0))
    brain_mask_casted.SetSpacing((0.9375, 0.9375, 3))

    #Apply bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #corrector.SetMaximumNumberOfIterations(num_iterations*number_fitting_levels)
    print("Correcting FLAIR...")
    correct_origin_and_spacing(flair_img_casted)
    flair_img_corrected = corrector.Execute(flair_img_casted, brain_mask_casted)
    print("Correcting T1_c...")
    correct_origin_and_spacing(t1c_img_casted)
    t1c_img_corrected = corrector.Execute(t1c_img_casted, brain_mask_casted)
    print("Correcting T1...")
    correct_origin_and_spacing(t1_img_casted)
    t1_img_corrected = corrector.Execute(t1_img_casted, brain_mask_casted)
    print("Correcting PD...")
    correct_origin_and_spacing(pd_img_casted)
    pd_img_corrected = corrector.Execute(pd_img_casted, brain_mask_casted)
    print("Correcting T2...")
    correct_origin_and_spacing(t2_img_casted)
    t2_img_corrected = corrector.Execute(t2_img_casted, brain_mask_casted)

    #Copy metadata and save
    flair_img_corrected.CopyInformation(flair_img)
    t1c_img_corrected.CopyInformation(t1c_img)
    t1_img_corrected.CopyInformation(t1_img)
    pd_img_corrected.CopyInformation(pd_img)
    t2_img_corrected.CopyInformation(t2_img)
    sitk.WriteImage(flair_img_corrected, jp(path_data, case, "FLAIR_masked_n4.nii.gz"))
    sitk.WriteImage(t1c_img_corrected, jp(path_data, case, "T1_c_masked_n4.nii.gz"))
    sitk.WriteImage(t1_img_corrected, jp(path_data, case, "T1_masked_n4.nii.gz"))
    sitk.WriteImage(pd_img_corrected, jp(path_data, case, "PD_masked_n4.nii.gz"))
    sitk.WriteImage(t2_img_corrected, jp(path_data, case, "T2_masked_n4.nii.gz"))

sf = 345
