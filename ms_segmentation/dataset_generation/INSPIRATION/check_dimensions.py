import os
import SimpleITK as sitk
from os.path import join as jp
import pandas as pd
from general.general import remove_from_list_if_contains


path_data = r"D:\dev"
original_data = 'INSPIRATION-org'
processed_data = 'INSPIRATION'

data = processed_data

#List all centers
study_centers_list = os.listdir(jp(path_data, data))



#Iterate through every study center
columns = ['SC','P','TP','M', 'T', 'S']
df = pd.DataFrame(columns = columns)
i=0
for study_center in study_centers_list:
    print("SC: ", study_center)
    patients_list = os.listdir(jp(path_data, data, study_center))
    patients_list = patients_list[:-1] #Remove last element that corresponds to reference or calibration image
    for patient in patients_list: #For every patient
        timepoints_list = os.listdir(jp(path_data, data, study_center, patient))
        for timepoint in timepoints_list: # For every time point
            modalities_list = os.listdir(jp(path_data, data, study_center, patient, timepoint))
            modalities_list = remove_from_list_if_contains(modalities_list, "null")
            for modality in modalities_list: # For every modality
                types_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality)) #Mask or image
                for type_ in types_list:
                    images_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, type_))
                    #TODO: Check that there is only one image in the folders
                    curr_img = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, type_, images_list[0]))
                    if 'FLAIR' in modality:
                        df.loc[i] = [study_center, patient, timepoint, modality, type_, curr_img.GetSize()]
                        #print(study_center, ', ', patient, ', ', timepoint, ', ', curr_img.GetSize())
                        i+=1

df.to_csv(path_data+"/test.csv")

'''
#Iterate through every study center
columns = ['SC','P','TP','M','S']
df = pd.DataFrame(columns = columns)
i=0
for study_center in study_centers_list:
    print("SC: ", study_center)
    patients_list = os.listdir(jp(path_data, data, study_center))
    patients_list = patients_list[:-1] #Remove last element that corresponds to reference or calibration image
    for patient in patients_list: #For every patient
        timepoints_list = os.listdir(jp(path_data, data, study_center, patient))
        for timepoint in timepoints_list: # For every time point
            modalities_list = os.listdir(jp(path_data, data, study_center, patient, timepoint))
            modalities_list = remove_from_list_if_contains(modalities_list, "null")
            for modality in modalities_list: # For every modality
                images_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality))
                #TODO: Check that there is only one image in the folders
                curr_img = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, images_list[0]))
                if 'FLAIR' in modality:
                    df.loc[i] = [study_center, patient, timepoint, modality, curr_img.GetSize()]
                    #print(study_center, ', ', patient, ', ', timepoint, ', ', curr_img.GetSize())
                    i+=1

df.to_csv(path_data+"/test.csv")
'''