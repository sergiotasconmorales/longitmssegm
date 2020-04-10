import os
import numpy as np
from os.path import join as jp
import SimpleITK as sitk 
from general.general import list_folders

path_data = r'D:\dev\ISBI_CS'
cases_list = list_folders(path_data)
for case in cases_list:
    print("Current case: ", case)
    flair_itk = sitk.ReadImage(jp(path_data, case, "flair.nii.gz"))
    flair_np = sitk.GetArrayFromImage(flair_itk)
    mask = flair_np > 0
    mask_itk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_itk.CopyInformation(flair_itk)
    sitk.WriteImage(mask_itk, jp(path_data, case, "brain_mask.nii.gz"))