# Ensemble of different models

import nibabel as nib
import numpy as np 
import os
from os.path import join as jp
from ms_segmentation.general.general import save_image, list_folders, create_folder

path_base = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation'
ensemble_name = 'aensemble2'
create_folder(jp(path_base, ensemble_name))
indexes_experiments = [13,14,15]
all_experiments = list_folders(path_base)
to_consider = [all_experiments[i] for i in indexes_experiments]
num_folds = 5
tp_amounts = []

data = dict().fromkeys(to_consider, None)
for k in data.keys():
    data[k] = []
# read images:
for exp in to_consider: # for each of the selected experiments
    for f in range(num_folds): # for each fold
        timepoints = []
        images = os.listdir(jp(path_base, exp, "fold" + str(f+1).zfill(2), "results", str(f+1).zfill(2))) # List all images for current fold
        for img in images:
            timepoints.append(nib.load(jp(path_base, exp, "fold" + str(f+1).zfill(2), "results", str(f+1).zfill(2), img)).get_fdata().astype(np.uint8))
        data[exp].append(timepoints)
        tp_amounts.append(len(images))
    
for f in range(num_folds):
    #Create folders for results
    create_folder(jp(path_base, ensemble_name, "fold" + str(f+1).zfill(2)))
    create_folder(jp(path_base, ensemble_name, "fold" + str(f+1).zfill(2), "results"))
    create_folder(jp(path_base, ensemble_name, "fold" + str(f+1).zfill(2), "results", str(f+1).zfill(2)))

# do ensembling
for f in range(num_folds): # for each fold
    for tp in range(tp_amounts[f]):
        to_ensemble = [data[exp][f][tp] for exp in to_consider] # to ensemble
        to_ensemble = np.stack(to_ensemble, axis = 0)
        sumi = np.sum(to_ensemble, axis = 0)
        res = sumi>= np.ceil(len(to_consider)/2)
        save_image(res.astype(np.uint8), jp(path_base, ensemble_name, "fold" + str(f+1).zfill(2), "results", str(f+1).zfill(2), str(f+1).zfill(2) + "_" + str(tp+1).zfill(2) + "_segm.nii.gz"))


