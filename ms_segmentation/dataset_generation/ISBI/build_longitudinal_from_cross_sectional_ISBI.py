import os
import shutil
from ms_segmentation.general.general import list_folders, create_folder, cls
from os.path import join as jp

origin = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\histogram_matched'
destiny = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\histogram_matched'

to_copy = ['flair', 'pd', 'mprage', 't2', 'brain_mask', 'mask1', 'mask2']

patients = list_folders(origin)
for pat in patients:
    create_folder(jp(destiny, pat))

    #Copy all timepoints from origin to destiny
    timepoints = list_folders(jp(origin, pat))
    for tp in timepoints:
        cls()
        print("Patient: ", pat)
        print("TP: ", tp)
        for elem in to_copy:
            shutil.copyfile(jp(origin, pat, tp, elem +'.nii.gz'),
                    jp(destiny, pat, elem + '_' + tp + '.nii.gz'))