import os 
import nibabel as nib
import numpy as np
from os.path import join as jp
from ms_segmentation.general.general import save_image, list_folders
from ms_segmentation.data_generation.patch_manager_3d import normalize_data

path_base = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\isbi_cs'
patients = list_folders(path_base)
for pat in patients:
    print("Patient ", pat)
    tps = list_folders(jp(path_base, pat))
    for tp in tps:
        print("TP: ", tp)
        flair = normalize_data(nib.load(jp(path_base, pat, tp, 'flair.nii.gz')).get_fdata())
        t2 = normalize_data(nib.load(jp(path_base, pat, tp, 't2.nii.gz')).get_fdata())
        pd = normalize_data(nib.load(jp(path_base, pat, tp, 'pd.nii.gz')).get_fdata())
        t1 = normalize_data(nib.load(jp(path_base, pat, tp, 'mprage.nii.gz')).get_fdata())
        t1_inv = normalize_data(np.max(t1) - t1)

        t2_flair = normalize_data(flair*t2)
        pd_flair = normalize_data(flair*pd)
        t1_inv_flair = normalize_data(flair*t1_inv)
        what = normalize_data(flair*(t2+pd+t1_inv))

        save_image(t2_flair, jp(path_base, pat, tp, "t2_times_flair.nii.gz"), orientation= "RAI")
        save_image(pd_flair, jp(path_base, pat, tp, "pd_times_flair.nii.gz"), orientation= "RAI")
        save_image(t1_inv_flair, jp(path_base, pat, tp, "t1_inv_times_flair.nii.gz"), orientation= "RAI")
        save_image(what, jp(path_base, pat, tp, "sum_times_flair.nii.gz"), orientation= "RAI")
