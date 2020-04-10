import os
from os.path import join as jp
import numpy as np
import SimpleITK as sitk 
from general.general import list_folders

path_data = r'D:\dev\ISBI_CS'
cases_list = list_folders(path_data)
for case in cases_list:
    print("Current case: ", case)
    files_list = os.listdir(jp(path_data, case))
    for file_ in files_list:
        if "gz" in file_:
            continue
        # Read image
        img_itk = sitk.ReadImage(jp(path_data, case, file_))
        # Save with new format
        sitk.WriteImage(img_itk, jp(path_data, case, file_ + ".gz"))
        # Build brain image from FLAIR image (simple thresholding)
        if "flair" in file_ and "brain_mask.nii.gz" not in files_list:
            flair_np = sitk.GetArrayFromImage(img_itk)
            mask = flair_np > 0
            mask_itk = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_itk.CopyInformation(img_itk)
            sitk.WriteImage(mask_itk, jp(path_data, case, "brain_mask.nii.gz"))

        os.remove(jp(path_data, case, file_))