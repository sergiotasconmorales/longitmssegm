import os
from os.path import join as jp
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk 
from ms_segmentation.general.general import list_folders, list_files_with_name_containing, create_folder

path_masks = r'D:\dev\s.tasconmorales\test_segm\PROCESS\INPUT\mri'
path_target = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\isbi_train'
path_copy = r'D:\dev\s.tasconmorales\wm_gm_masks'

patients = list_folders(path_masks)

for pat in patients:
    print('Patient:', pat) 
    create_folder(jp(path_copy, pat))
    n_timepoints = len(list_files_with_name_containing(jp(path_target, pat), 'mprage', 'nii.gz'))
    #read ref image
    ref = sitk.ReadImage(jp(path_masks, pat, 'ref.nii.gz'))
    #Read each mask
    for tp in range(n_timepoints):
        print('Time-point:', tp+1)
        gm = sitk.ReadImage(jp(path_masks, pat, 'p1mprage_' + str(tp+1).zfill(2) + '.nii'))
        gm_np = sitk.GetArrayFromImage(gm)
        gm_new = sitk.GetImageFromArray((gm_np>0.5).astype(np.uint8))
        gm_new.CopyInformation(ref)
        sitk.WriteImage(gm_new, jp(path_target, pat, 'gm_' + str(tp+1).zfill(2) + '.nii.gz'))
        sitk.WriteImage(gm_new, jp(path_copy, pat, 'gm_' + str(tp+1).zfill(2) + '.nii'))
        wm = sitk.ReadImage(jp(path_masks, pat, 'p2mprage_' + str(tp+1).zfill(2) + '.nii'))
        wm_np = sitk.GetArrayFromImage(wm)
        wm_new = sitk.GetImageFromArray((wm_np>=0.5).astype(np.uint8))
        wm_new.CopyInformation(ref)
        sitk.WriteImage(wm_new, jp(path_target, pat, 'wm_' + str(tp+1).zfill(2) + '.nii.gz'))
        sitk.WriteImage(wm_new, jp(path_copy, pat, 'wm_' + str(tp+1).zfill(2) + '.nii.gz'))