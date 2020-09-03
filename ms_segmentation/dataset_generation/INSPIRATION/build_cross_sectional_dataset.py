## CROSS-SECTIONAL DATASET CREATION FROM INSPIRATION DATASET

import os
import SimpleITK as sitk
from os.path import join as jp
import pandas as pd
from shutil import copy2, copy
from general.general import remove_from_list_if_contains, create_folder, find_element_in_list_that_contains, print_line

path_data = r"D:\dev"
original_data = 'INSPIRATION-org'
processed_data = 'INSPIRATION'
new_data = 'CROSS_SECTIONAL'

identifier = 0
discarded_counter = 0

#Create dataset folder if it doesn't exist
create_folder(jp(path_data, new_data))


columns = ['StudyCenter','Patient','TimePoint', 'Identifier']
df = pd.DataFrame(columns = columns)

columns2 = ['StudyCenter','Patient','TimePoint', 'Reason']
df2 = pd.DataFrame(columns = columns2)

data = processed_data
study_centers_list = os.listdir(jp(path_data, data))
sc_cnt = 1
#Iterate through every study center
for study_center in study_centers_list:
    print_line()
    print("Study Center: ", study_center, " -> ", sc_cnt, "/", len(study_centers_list))
    print_line()
    patients_list = os.listdir(jp(path_data, data, study_center))
    patients_list = patients_list[:-1] #Remove last element that corresponds to reference or calibration image
    p_cnt = 1
    for patient in patients_list: #For every patient
        print("Patient: ", patient, " -> ", p_cnt, "/", len(patients_list))
        timepoints_list = os.listdir(jp(path_data, data, study_center, patient))
        timepoints_list_orig = os.listdir(jp(path_data, original_data, study_center, patient))
        if(len(timepoints_list)!=len(timepoints_list_orig)): #For SC 0012 sometimes doesnt agree
            print("INVALID - NUMBER OF TIMEPOINTS MISMATCH")
            df2.loc[discarded_counter] = [study_center, patient, "ALL", "NUMBER OF TIMEPOINTS MISMATCH"]
            discarded_counter += 1
            continue
        for i_timepoint in range(len(timepoints_list)): # For every time point
            timepoint = timepoints_list[i_timepoint]
            modalities_list = os.listdir(jp(path_data, data, study_center, patient, timepoint))
            modalities_list = remove_from_list_if_contains(modalities_list, "null")
            for modality in modalities_list: # For every modality
                if modality == "FLAIR": #If FLAIR 2D
                    types_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality)) #There should be only mask folder
                    images_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, types_list[0]))
                    curr_flair_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, types_list[0], images_list[0])) #Take first image which is 3D volume (MASK9)
                    #Correct masks that have more than 2 possible labels
                    curr_flair_proc = curr_flair_proc > 0
                    
                    timepoints_orig = os.listdir(jp(path_data, original_data, study_center, patient))
                    if not os.path.exists(jp(path_data, original_data, study_center, patient, timepoints_orig[i_timepoint],'FLAIR_T2')): #If there is no FLAIR, jump
                        print("INVALID - NO FLAIR2D IN ORIGINAL DATA")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "NO FLAIR2D IN ORIGINAL DATA"]
                        discarded_counter += 1
                        break
                    flair_orig_name = os.listdir(jp(path_data, original_data, study_center, patient, timepoints_orig[i_timepoint],'FLAIR_T2'))
                    curr_flair_orig = sitk.ReadImage(jp(path_data, original_data, study_center, patient, timepoints_orig[i_timepoint], "FLAIR_T2", flair_orig_name[0]))
                    

                    t1c_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native'))
                    curr_t1c_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native', t1c_proc_name[0]))

                    t1_types = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1'))
                    if not len(t1_types)==2:
                        raise ValueError("Error with " + str(study_center) + ", patient " + str(patient) + ", timepoint " + str(timepoint) + ". T1 image does not contain both images and masks")
                    t1_img_folder_name = find_element_in_list_that_contains(t1_types, "RMI")
                    t1_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name))
                    curr_t1_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name, t1_proc_name[0]))
                    
                    t1_mask_folder_name = find_element_in_list_that_contains(t1_types, "LES")
                    t1_mask_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name))
                    curr_t1_mask_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name, t1_mask_name[0]))
                    curr_t1_mask_proc = curr_t1_mask_proc > 0

                    if(curr_flair_proc.GetSize() == curr_flair_orig.GetSize() == curr_t1c_proc.GetSize() == curr_t1_proc.GetSize()):
                        print("VALID")
                        curr_flair_proc.CopyInformation(curr_t1c_proc)
                        curr_flair_orig.CopyInformation(curr_t1c_proc)
                        curr_t1_proc.CopyInformation(curr_t1c_proc)
                        curr_t1_mask_proc.CopyInformation(curr_t1c_proc)

                        new_folder_name = str(identifier+1).zfill(5)
                        case_folder_name = jp(path_data, new_data, new_folder_name)
                        create_folder(case_folder_name)
                        #Copy files to created folder
                        #Copy T1:
                        sitk.WriteImage(curr_t1_proc, jp(path_data, new_data, new_folder_name, 'T1.nii.gz'))
                        #Copy T1c_
                        copy2(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native', t1c_proc_name[0]), jp(path_data, new_data, new_folder_name, 'T1_c.nii.gz'))
                        #Write FLAIR:
                        sitk.WriteImage(curr_flair_orig, jp(path_data, new_data, new_folder_name, 'FLAIR.nii.gz'))
                        #Copy mask FLAIR:
                        sitk.WriteImage(curr_flair_proc, jp(path_data, new_data, new_folder_name, 'mask_FLAIR.nii.gz'))
                        #copy2(jp(path_data, data, study_center, patient, timepoint, modality, types_list[0], images_list[0]), jp(path_data, new_data, new_folder_name, 'mask_FLAIR.nii.gz'))
                        #Copy mask T1:
                        sitk.WriteImage(curr_t1_mask_proc, jp(path_data, new_data, new_folder_name, 'mask_T1.nii.gz'))

                        df.loc[identifier] = [study_center, patient, timepoint, str(identifier+1).zfill(5)]

                        identifier += 1
                    else:
                        print("INVALID - DIMENSIONS MISMATCH")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "DIMENSIONS MISMATCH"]
                        discarded_counter += 1

                elif modality == '3DFLAIR': # If FLAIR 3D. In this case I can take all images from processed data

                    #First, read all images and check dimensions
                    #FLAIR
                    flair_types_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality)) #Should be two folders: Image and mask
                    if not len(flair_types_list)>1:
                        raise ValueError("Error with " + str(study_center) + ", patient " + str(patient) + ", timepoint " + str(timepoint) + ". 3DFLAIR image does not contain both images and masks")
                    curr_flair_proc_folder_name = find_element_in_list_that_contains(flair_types_list, "FLAIR")
                    curr_flair_proc_img_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_proc_folder_name))[0]
                    curr_flair_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DFLAIR', curr_flair_proc_folder_name, curr_flair_proc_img_name))
                    curr_flair_mask_proc_folder_name = find_element_in_list_that_contains(flair_types_list, "LES")
                    curr_flair_proc_mask_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_mask_proc_folder_name))[0]
                    curr_flair_mask = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_mask_proc_folder_name, curr_flair_proc_mask_name))
                    curr_flair_mask = curr_flair_mask > 0 # Correct in case more than two labels

                    #T1_c
                    t1c_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native'))
                    curr_t1c_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native', t1c_proc_name[0]))

                    t1_types = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1'))
                    if not len(t1_types)==2:
                        raise ValueError("Error with " + str(study_center) + ", patient " + str(patient) + ", timepoint " + str(timepoint) + ". T1 image does not contain both images and masks")
                    t1_img_folder_name = find_element_in_list_that_contains(t1_types, "RMI")
                    t1_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name))
                    curr_t1_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name, t1_proc_name[0]))
                    
                    t1_mask_folder_name = find_element_in_list_that_contains(t1_types, "LES")
                    t1_mask_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name))
                    curr_t1_mask_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name, t1_mask_name[0]))
                    curr_t1_mask_proc = curr_t1_mask_proc > 0

                    if(curr_flair_proc.GetSize() == curr_flair_mask.GetSize() == curr_t1c_proc.GetSize() == curr_t1_proc.GetSize()):
                        print("VALID")
                        new_folder_name = str(identifier+1).zfill(5)
                        case_folder_name = jp(path_data, new_data, new_folder_name)
                        create_folder(case_folder_name)
                        #Copy FLAIR    
                        copy2(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_proc_folder_name, curr_flair_proc_img_name), jp(path_data, new_data, new_folder_name, 'FLAIR.nii.gz'))
                        #Write FLAIR mask
                        sitk.WriteImage(curr_flair_mask, jp(path_data, new_data, new_folder_name, 'mask_FLAIR.nii.gz'))
                        #Copy T1_c
                        copy2(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native', t1c_proc_name[0]), jp(path_data, new_data, new_folder_name, 'T1_c.nii.gz'))
                        #Copy T1
                        copy2(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name, t1_proc_name[0]), jp(path_data, new_data, new_folder_name, 'T1.nii.gz'))
                        #Copy T1 mask
                        sitk.WriteImage(curr_t1_mask_proc, jp(path_data, new_data, new_folder_name, 'mask_T1.nii.gz'))

                        df.loc[identifier] = [study_center, patient, timepoint, str(identifier+1).zfill(5)]
                        identifier += 1
                    else:
                        print("INVALID - DIMENSIONS MISMATCH")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "DIMENSIONS MISMATCH"]
                        discarded_counter += 1
                else:
                    pass
        p_cnt += 1            
    sc_cnt += 1

df.to_csv(jp(path_data, new_data, "assignation.csv"))
df2.to_csv(jp(path_data, new_data, "discarded.csv"))