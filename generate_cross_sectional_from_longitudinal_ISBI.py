import os
from os.path import join as jp
import shutil
from ms_segmentation.general.general import list_folders, create_folder

path_origin = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\chi_square_images_longit'
path_destiny = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\chi_square_images_cs'
create_folder(path_destiny)
patients = list_folders(path_origin)

def get_name(the_string):
    if "flair" in the_string:
        return "flair_norm.nii.gz"
    elif "brain_mask" in the_string:
        return "brain_mask.nii.gz"
    elif "pd" in the_string:
        return "pd_norm.nii.gz"
    elif "mprage" in the_string:
        return "mprage_norm.nii.gz"
    elif "t2" in the_string:
        return "t2_norm.nii.gz"
    elif "mask1" in the_string:
        return "mask1.nii.gz"
    elif "mask2" in the_string:
        return "mask2.nii.gz"
    else:
        raise Exception("Unknown image type")

for pat in patients:
    create_folder(jp(path_destiny, pat))
    all_files = os.listdir(jp(path_origin, pat))
    num_tp = int(len(all_files)/5) # 7 because there are 7 types of images
    for tp in range(num_tp):
        create_folder(jp(path_destiny, pat, str(tp+1).zfill(2)))
        for f in all_files:
            if str(tp+1).zfill(2) in f: # If number of timepoint present, copy file
                shutil.copyfile(jp(path_origin, pat, f), jp(path_destiny, pat, str(tp+1).zfill(2), get_name(f)))



