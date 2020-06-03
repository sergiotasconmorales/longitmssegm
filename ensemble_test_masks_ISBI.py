import os
import numpy as np
import nibabel as nib
from os.path import join as jp
from ms_segmentation.general.general import save_image, list_folders, create_log, create_folder, list_files_with_extension


path_base = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs'
path_ensembles = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs\ensembles'
experiments_to_ensemble = ['qwertz06', 
                            'qwertz09',
                            'qwertz11']

num_ens = len(list_folders(path_ensembles))
name_folder_new_ensemble = "Ensemble_" + str(num_ens+1).zfill(2)
create_folder(jp(path_ensembles, name_folder_new_ensemble))

# Check that listed experiments exist
all_experiments = list_folders(path_base)
all_experiments = [x for x in all_experiments if x in experiments_to_ensemble]

assert len(all_experiments) % 2 != 0 # number of experiments must be odd

all_images = []
for i_exp, exp in enumerate(all_experiments):
    print("Current experiment: ", exp, i_exp+1, "/", len(all_experiments))
    all_images_names = list_files_with_extension(jp(path_base, exp), "nii")
    all_images_curr_exp = []
    for img in all_images_names:
        all_images_curr_exp.append(nib.load(jp(path_base, exp, img)).get_fdata().astype(np.uint8))
    all_images.append(np.stack(all_images_curr_exp)) # (61, volume size)

all_images_np = np.stack(all_images)
the_sum = np.sum(all_images_np, axis = 0)
voting = the_sum >= np.ceil(len(all_experiments)/2)

# save images
for i_case in range(voting.shape[0]):
    save_image(voting[i_case,:,:,:].astype(np.uint8), jp(path_ensembles, name_folder_new_ensemble, all_images_names[i_case]))

create_log(jp(path_ensembles, name_folder_new_ensemble), {'experiments': all_experiments})