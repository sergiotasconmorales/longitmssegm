## Script to rename product images that contain "flair" in their names so that when listing files they are not liste
# twice.

import os
from os.path import join as jp
from ms_segmentation.general.general import list_folders, list_files_with_name_containing

path_images = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\isbi_train'
patients = list_folders(path_images)

changes = {'pd_times_flair': 'pd_times_fl', 't2_times_flair': 't2_times_fl', 't1_inv_times_flair': 't1_inv_times_fl', 'sum_times_flair': 'sum_times_fl'}

for pat in patients: # for each patient
    print("patient: ", pat)
    for k,v in changes.items():
        list_of_files = list_files_with_name_containing(jp(path_images, pat), k, "nii.gz")
        for i_tp in range(len(list_of_files)): # for each timepoint
            print("timepoint: ", i_tp+1)
            new_name = list_of_files[i_tp].replace("flair", "fl")
            os.rename(list_of_files[i_tp], new_name)
            


