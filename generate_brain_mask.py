import os
import numpy as np
from os.path import join as jp
import SimpleITK as sitk 
from ms_segmentation.general.general import list_folders

path_data = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\cross_sectional'
cases_list = list_folders(path_data)
for case in cases_list:
    print("Current case: ", case)
    tps = list_folders(jp(path_data, case))
    for tp in tps:
        flair_itk = sitk.ReadImage(jp(path_data, case, tp, "flair.nii.gz"))
        flair_np = sitk.GetArrayFromImage(flair_itk)
        mask = flair_np > 0
        mask_itk = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_itk.CopyInformation(flair_itk)
        sitk.WriteImage(mask_itk, jp(path_data, case, tp, "brain_mask.nii.gz"))