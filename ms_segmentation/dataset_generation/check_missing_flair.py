import os
import SimpleITK as sitk
from os.path import join as jp
import pandas as pd
from general.general import remove_from_list_if_contains


action = 3
#1: check missing flair in original dataset
#2: check number of timepoints per patient

path_data = r"D:\dev"
original_data = 'INSPIRATION-org'
processed_data = 'INSPIRATION'






if action==1:
    data = original_data
    study_centers_list = os.listdir(jp(path_data, data))
    #Iterate through every study center
    missing_count = 0
    for study_center in study_centers_list:
        #print("SC: ", study_center)
        patients_list = os.listdir(jp(path_data, data, study_center))
        patients_list = patients_list[:-1] #Remove last element that corresponds to reference or calibration image
        for patient in patients_list: #For every patient
            timepoints_list = os.listdir(jp(path_data, data, study_center, patient))
            for timepoint in timepoints_list: # For every time point
                modalities_list = os.listdir(jp(path_data, data, study_center, patient, timepoint))
                modalities_list = remove_from_list_if_contains(modalities_list, "null") #Exclude "null" folders
                if 'FLAIR_T2' in modalities_list:
                    print(study_center, ', ', patient, ', ', timepoint, ', ', 'OK')
                else:
                    print(study_center, ', ', patient, ', ', timepoint, ', ', 'MISSING')
                    missing_count += 1
                            
    print("Missing: ", missing_count)

elif action==2:
    data = processed_data
    study_centers_list = os.listdir(jp(path_data, data))
    columns = ['SC','P','nTP']
    df = pd.DataFrame(columns = columns)
    i=0
    for study_center in study_centers_list:
        print("SC: ", study_center)
        patients_list = os.listdir(jp(path_data, data, study_center))
        patients_list = patients_list[:-1] #Remove last element that corresponds to reference or calibration image
        for patient in patients_list: #For every patient
            timepoints_list = os.listdir(jp(path_data, data, study_center, patient))
            df.loc[i] = [study_center, patient, len(timepoints_list)]
            #print(study_center, ', ', patient, ', ', timepoint, ', ', curr_img.GetSize())
            i+=1

    df.to_csv(path_data+"/num_timepoints_per_patient.csv")

elif action==3:
    data = processed_data
    study_centers_list = os.listdir(jp(path_data, data))
    #Iterate through every study center
    for study_center in study_centers_list:
        #print("SC: ", study_center)
        patients_list = os.listdir(jp(path_data, data, study_center))
        patients_list = patients_list[:-1] #Remove last element that corresponds to reference or calibration image
        for patient in patients_list: #For every patient
            timepoints_list = os.listdir(jp(path_data, data, study_center, patient))
            timepoints_list_orig = os.listdir(jp(path_data, original_data, study_center, patient))
            if(len(timepoints_list)!=len(timepoints_list_orig)): #For SC 0012 sometimes doesnt agree
                continue
            for i_timepoint in range(len(timepoints_list)): # For every time point
                timepoint = timepoints_list[i_timepoint]
                modalities_list = os.listdir(jp(path_data, data, study_center, patient, timepoint))
                modalities_list = remove_from_list_if_contains(modalities_list, "null")
                for modality in modalities_list: # For every modality
                    if modality == "FLAIR": #If FLAIR 2D
                        types_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality)) #There is only mask folder
                        images_list = os.listdir(jp(path_data, data, study_center, patient, timepoint, modality, types_list[0]))
                        curr_flair_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, modality, types_list[0], images_list[0])) #Take first image which is 3D volume
                        timepoints_orig = os.listdir(jp(path_data, original_data, study_center, patient))
                        if not os.path.exists(jp(path_data, original_data, study_center, patient, timepoints_orig[i_timepoint],'FLAIR_T2')):
                            break
                        flair_orig_name = os.listdir(jp(path_data, original_data, study_center, patient, timepoints_orig[i_timepoint],'FLAIR_T2'))
                        curr_flair_orig = sitk.ReadImage(jp(path_data, original_data, study_center, patient, timepoints_orig[i_timepoint], "FLAIR_T2", flair_orig_name[0]))

                        t1c_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native'))
                        curr_t1c_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, 'T1Gd', 'Native', t1c_proc_name[0]))

                        t1_proc_name = os.listdir(jp(path_data, data, study_center, patient, timepoint, '3DT1', 'RMI_T1_3mm'))
                        curr_t1_proc = sitk.ReadImage(jp(path_data, data, study_center, patient, timepoint, '3DT1', 'RMI_T1_3mm', t1_proc_name[0]))

                        print(study_center, ', ', patient, ', ', timepoint, ', ', modality, ', ',  curr_flair_proc.GetSize() == curr_flair_orig.GetSize() == curr_t1c_proc.GetSize() == curr_t1_proc.GetSize())
                    else:
                        continue
                    

