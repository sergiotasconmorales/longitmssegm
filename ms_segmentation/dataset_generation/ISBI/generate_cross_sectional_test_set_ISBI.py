import numpy as np 
import nibabel as nib 
import os
import shutil
from os.path import join as jp
from ms_segmentation.general.general import list_folders, list_files_with_name_containing, create_folder

path_base = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images'
path_cs = jp(path_base, 'cross_sectional')
path_origin = jp(path_base, 'orig')
patients = list_folders(path_origin)

case_index = 1

for i_pat, pat in enumerate(patients):
    print("Patient: ", i_pat+1, "/", len(patients))
    path_pat = jp(path_origin, pat, "preprocessed")
    flair_images = list_files_with_name_containing(path_pat, "flair", "nii")
    mprage_images = list_files_with_name_containing(path_pat, "mprage", "nii")
    pd_images = list_files_with_name_containing(path_pat, "pd", "nii")
    t2_images = list_files_with_name_containing(path_pat, "t2", "nii")

    #check that all modalities have the same number of images
    assert len(flair_images) == len(mprage_images) == len(pd_images) == len(t2_images)

    #create patient folder
    create_folder(jp(path_cs, str(case_index).zfill(2)))

    for i_img in range(len(flair_images)):
        print("Timepoint: ", i_img+1)
        # create timepoint folder
        create_folder(jp(path_cs, str(case_index).zfill(2), str(i_img+1).zfill(2)))
        shutil.copyfile(flair_images[i_img], jp(path_cs, str(case_index).zfill(2), str(i_img+1).zfill(2), "flair.nii"))
        shutil.copyfile(mprage_images[i_img], jp(path_cs, str(case_index).zfill(2), str(i_img+1).zfill(2), "mprage.nii"))
        shutil.copyfile(pd_images[i_img], jp(path_cs, str(case_index).zfill(2), str(i_img+1).zfill(2), "pd.nii"))
        shutil.copyfile(t2_images[i_img], jp(path_cs, str(case_index).zfill(2), str(i_img+1).zfill(2), "t2.nii"))
    
    case_index += 1

#change extension
for i_pat, pat in enumerate(patients):







