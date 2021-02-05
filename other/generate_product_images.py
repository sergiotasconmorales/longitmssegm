import os 
import nibabel as nib
import numpy as np
from os.path import join as jp
from ms_segmentation.general.general import save_image, list_folders, list_files_with_name_containing
from ms_segmentation.data_generation.patch_manager_3d import normalize_data

path_base = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\cross_sectional'
cross_sectional = True

patients = list_folders(path_base)
if cross_sectional:
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
else:
    for pat in patients:
        print("Patient ", pat)
        flair_files = list_files_with_name_containing(jp(path_base, pat), "flair", "nii.gz")
        t2_files = list_files_with_name_containing(jp(path_base, pat), "t2", "nii.gz")
        mprage_files = list_files_with_name_containing(jp(path_base, pat), "mprage", "nii.gz")
        pd_files = list_files_with_name_containing(jp(path_base, pat), "pd", "nii.gz")
        assert len(flair_files) == len(t2_files) == len(mprage_files) == len(pd_files), "Modality missing for some timepoint"
        for i in range(len(flair_files)):
            print("TP: ", i)
            flair = normalize_data(nib.load(jp(flair_files[i])).get_fdata())
            t2 = normalize_data(nib.load(jp(t2_files[i])).get_fdata())
            pd = normalize_data(nib.load(jp(pd_files[i])).get_fdata())
            t1 = normalize_data(nib.load(jp(mprage_files[i])).get_fdata())
            t1_inv = normalize_data(np.max(t1) - t1)

            t2_flair = normalize_data(flair*t2)
            pd_flair = normalize_data(flair*pd)
            t1_inv_flair = normalize_data(flair*t1_inv)
            what = normalize_data(flair*(t2+pd+t1_inv))

            save_image(t2_flair, jp(path_base, pat, "t2_times_flair_"+ str(i+1).zfill(2) + ".nii.gz"), orientation= "RAI")
            save_image(pd_flair, jp(path_base, pat, "pd_times_flair_"+ str(i+1).zfill(2) +".nii.gz"), orientation= "RAI")
            save_image(t1_inv_flair, jp(path_base, pat, "t1_inv_times_flair_"+ str(i+1).zfill(2) +".nii.gz"), orientation= "RAI")
            save_image(what, jp(path_base, pat, "sum_times_flair_"+ str(i+1).zfill(2) + ".nii.gz"), orientation= "RAI")
