import os
import SimpleITK as sitk
from os.path import join as jp
from .general.general import create_folder
from shutil import copy2


path_origin = r'X:\ST\FLAIR\INSPIRATION_Final_FLAIR' #Path to study centers
path_destiny = r'D:\dev\INSPIRATION' 

centers_list = os.listdir(path_origin)

for center in centers_list:
    print("Center: ", center)
    patients_list = os.listdir(jp(path_origin, center))
    for patient in patients_list[:-1]:
        timepoints_list = os.listdir(jp(path_origin, center, patient))
        for timepoint in timepoints_list:
            #Create folder in destini
            create_folder(jp(path_destiny, center, patient, timepoint, "FLAIR", "Native"))
            files_list = os.listdir(jp(path_origin, center, patient, timepoint, "FLAIR", "Native"))
            for file in files_list:
                #Copy every file to corresponding folder
                copy2(jp(path_origin, center, patient, timepoint, "FLAIR", "Native", file), jp(path_destiny, center, patient, timepoint, "FLAIR", "Native", file))