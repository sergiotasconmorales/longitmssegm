import os
from os.path import join as jp
from shutil import copy
from ms_segmentation.general.general import list_folders, list_files_with_name_containing
import shutil

path_origin = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\longitudinal'
path_target = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\chi_square_images_longit'

to_copy = ["brain_mask"]

patients = list_folders(path_origin)
for pat in patients:
    files_to_copy = [list_files_with_name_containing(jp(path_origin, pat), stri, "nii.gz") for stri in to_copy]
    for t in range(len(to_copy)):
        files = files_to_copy[t]
        for f in range(len(files)):
            shutil.copyfile(files[f], jp(path_target, pat, to_copy[t] + "_" + str(f+1).zfill(2) +".nii.gz"))