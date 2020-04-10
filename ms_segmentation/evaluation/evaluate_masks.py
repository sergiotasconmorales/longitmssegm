# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script for evaluation of segmentation masks in a comparison with the ground truth 
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------

import os
import nibabel as nib
import numpy as np
from ..general.general import list_folders
from ..evaluation.metrics import compute_dices
from os.path import join as jp

path_gt = r'D:\dev\INS_Test\test' #Folders
path_results = r'D:\dev\INS_Test\results' #All together

#By convention, segmentation masks are called <case>_segm.nii.gz

all_dices = []

#For each case in GT, compare
cases_gt = list_folders(path_gt)
for case in cases_gt:

    if not os.path.exists(jp(path_results, case)):
        continue
    #Read GT
    labels_gt = nib.load(jp(path_gt, case, "mask_FLAIR.nii.gz")).get_fdata().astype(np.uint8)
    #Read segmentation result
    labels_segm = nib.load(jp(path_results, case, case + "_segm.nii.gz")).get_fdata().astype(np.uint8)

    dsc = compute_dices(labels_gt.flatten(), labels_segm.flatten())
    print("Case: ", case, )
    print("     DSC: %6.4f" % (dsc[0]))

    all_dices.append(dsc[0])

mean_dsc = np.mean(np.array(all_dices))
std_dsc = np.std(np.array(all_dices))
print("Mean DSC: %6.4f (%6.4f)" % (mean_dsc, std_dsc))