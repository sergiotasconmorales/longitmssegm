## CROSS-SECTIONAL DATASET CREATION FROM INSPIRATION DATASET

import os
import SimpleITK as sitk
from os.path import join as jp
import pandas as pd
from shutil import copy2, copy
from general.general import remove_from_list_if_contains, create_folder, find_element_in_list_that_contains, print_line

path_data = r"D:\dev"
processed_data = 'INSPIRATION'
orig_data = 'INSPIRATION-org'
new_data = 'CROSS_SECTIONAL3'

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
        # timepoints_list_orig = os.listdir(jp(path_data, original_data, study_center, patient))
        # if(len(timepoints_list)!=len(timepoints_list_orig)): #For SC 0012 sometimes doesnt agree
        #     print("INVALID - NUMBER OF TIMEPOINTS MISMATCH")
        #     df2.loc[discarded_counter] = [study_center, patient, "ALL", "NUMBER OF TIMEPOINTS MISMATCH"]
        #     discarded_counter += 1
        #     continue
        for i_timepoint in range(len(timepoints_list)): # For every time point
            timepoint = timepoints_list[i_timepoint]
            modalities_list = os.listdir(jp(path_data, data, study_center, patient, timepoint))
            modalities_list = remove_from_list_if_contains(modalities_list, "null")
            if not (bool(find_element_in_list_that_contains(modalities_list, 'T1') and bool(find_element_in_list_that_contains(modalities_list, 'FLAIR') and bool(find_element_in_list_that_contains(modalities_list, 'Gd'))))):
                print("INVALID - NOT ALL 3 SEQUENCES PRESENT")
                df2.loc[discarded_counter] = [study_center, patient, timepoint, "NOT ALL 3 SEQUENCES PRESENT"]
                discarded_counter += 1
                break
            for modality in modalities_list: # For every modality
                if modality == "FLAIR": #If FLAIR 2D
                    #First, read all images and check dimensions
                    #FLAIR
                    flair_types_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality)) #Should be two folders: Image and mask
                    if not len(flair_types_list)>1:
                        print("INVALID - 2DFLAIR image does not contain both images and masks")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "ORIGINAL IMAGES OR MASKS MISSING FOR 2DFLAIR"]
                        discarded_counter += 1
                        break
                    curr_flair_proc_folder_name = find_element_in_list_that_contains(flair_types_list, "Native")
                    curr_flair_proc_img_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_proc_folder_name))[0]
                    curr_flair_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_proc_folder_name, curr_flair_proc_img_name))
                    curr_flair_mask_proc_folder_name = find_element_in_list_that_contains(flair_types_list, "LES")
                    curr_flair_proc_mask_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_mask_proc_folder_name))[0]
                    curr_flair_mask = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, curr_flair_mask_proc_folder_name, curr_flair_proc_mask_name))
                    curr_flair_mask = curr_flair_mask > 0 # Correct in case more than two labels

                    #T1_c
                    t1c_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native'))
                    curr_t1c_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native', t1c_proc_name[0]))

                    t1_types = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1'))
                    if not len(t1_types)>1:
                        print("INVALID - T1 image does not contain both images and masks")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "ORIGINAL IMAGES OR MASKS MISSING FOR T1"]
                        discarded_counter += 1
                        break
                    t1_img_folder_name = find_element_in_list_that_contains(t1_types, "RMI")
                    t1_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name))
                    curr_t1_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name, t1_proc_name[0]))
                    
                    t1_mask_folder_name = find_element_in_list_that_contains(t1_types, "LES")
                    t1_mask_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name))
                    curr_t1_mask_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name, t1_mask_name[0]))
                    curr_t1_mask_proc = curr_t1_mask_proc > 0

                    #Check that T2 and PD exist
                    if not os.path.exists(jp(path_data, data, study_center, patient, timepoint, 'PD', 'Native')) or not os.path.exists(jp(path_data, data, study_center, patient, timepoint, 'PD', 'Native')):
                        print("INVALID - T2 and PD not present")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "MISSING T2 AND PD"]
                        discarded_counter += 1
                        break


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
                        #Copy PD
                        copy2(jp(path_data, data, study_center, patient, timepoint, 'PD', 'Native', "PD.nii.gz"), jp(path_data, new_data, new_folder_name, 'PD.nii.gz'))
                        #Copy T2
                        copy2(jp(path_data, data, study_center, patient, timepoint, 'T2', 'Native', "T2.nii.gz"), jp(path_data, new_data, new_folder_name, 'T2.nii.gz'))

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
                        print("INVALID - 3DFLAIR image does not contain both images and masks")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "ORIGINAL IMAGES OR MASKS MISSING FOR 3DFLAIR"]
                        discarded_counter += 1
                        break
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
                    if not len(t1_types)>1:
                        print("INVALID - T1 image does not contain both images and masks")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "ORIGINAL IMAGES OR MASKS MISSING FOR T1"]
                        discarded_counter += 1
                        break
                    t1_img_folder_name = find_element_in_list_that_contains(t1_types, "RMI")
                    t1_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name))
                    curr_t1_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_img_folder_name, t1_proc_name[0]))
                    
                    t1_mask_folder_name = find_element_in_list_that_contains(t1_types, "LES")
                    t1_mask_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name))
                    curr_t1_mask_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', t1_mask_folder_name, t1_mask_name[0]))
                    curr_t1_mask_proc = curr_t1_mask_proc > 0

                    #Check that T2 and PD exist
                    if not os.path.exists(jp(path_data, data, study_center, patient, timepoint, 'PD', 'Native')) or not os.path.exists(jp(path_data, data, study_center, patient, timepoint, 'PD', 'Native')):
                        print("INVALID - T2 and PD not present")
                        df2.loc[discarded_counter] = [study_center, patient, timepoint, "MISSING T2 AND PD"]
                        discarded_counter += 1
                        break

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
                        #Copy PD
                        copy2(jp(path_data, data, study_center, patient, timepoint, 'PD', 'Native', "PD.nii.gz"), jp(path_data, new_data, new_folder_name, 'PD.nii.gz'))
                        #Copy T2
                        copy2(jp(path_data, data, study_center, patient, timepoint, 'T2', 'Native', "T2.nii.gz"), jp(path_data, new_data, new_folder_name, 'T2.nii.gz'))

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