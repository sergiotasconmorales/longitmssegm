import nibabel as nib
import os
import numpy as np
from os.path import join as jp
from ms_segmentation.data_fusion import vggfusion
from ms_segmentation.general.general import save_image, list_folders, cls
from ms_segmentation.data_generation.patch_manager_2d import normalize_data

path_images = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\isbi_cs'
patients = list_folders(path_images)

for pat in patients:
    timepoints = list_folders(jp(path_images, pat))
    for tp in timepoints:
        flair = normalize_data(nib.load(jp(path_images, pat, tp, "flair.nii.gz")).get_fdata(), norm_type='zero_one')
        t2 = normalize_data(nib.load(jp(path_images, pat, tp, "t2.nii.gz")).get_fdata(), norm_type='zero_one')
        fused = np.zeros_like(flair, dtype = np.float32)
        for i_slice in range(fused.shape[2]):
            cls()
            print("Patient:", pat, end="  ")
            print("Time point:", tp)
            print("Slice:", i_slice+1)
            fused[:,:,i_slice] = vggfusion.fuse(flair[:,:,i_slice], t2[:,:,i_slice])
        save_image(fused, jp(path_images, pat, tp, "fused_flt2.nii.gz"))



