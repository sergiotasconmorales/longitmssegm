import os
import SimpleITK as sitk
import numpy as np 
from os.path import join as jp
from general.general import create_folder, list_folders, list_files_with_extension, find_element_in_list_that_contains, cls

path_original = r'D:\dev\INSPIRATION-org'
path_destiny = r'D:\dev\INSPIRATION'

def get_pd_and_t2(combined):
    """Function to separate pd and t2 from dual image
    """
    th = 30

    img_np = sitk.GetArrayFromImage(combined)
    new_images = np.zeros((2, int(img_np.shape[0]/2), img_np.shape[1], img_np.shape[2]))

    
    first = np.mean(img_np[0,:,:])
    second = np.mean(img_np[1,:,:])
    third = np.mean(img_np[2,:,:])

    analyze_third = False

    if second > first: # If second slice brighter than first
        if second - first > th: 
            #Second is PD and first is T2
            init = "T2"
        else:
            #Same (analyze third)
            analyze_third = True
    else:
        if first - second > th:
            # First is PD and second is T2
            init = "PD"
        else:
            #Same (analyze third)
            analyze_third = True

    if analyze_third:
        if third > second:
            if third - second > th:
                #Third is PD, first two are T2
                init = "T2"
            else:
                #This shouldn't happen
                print("Three slices with same sequence")
        else:
            if second - third > th:
                # Third is T2, first two are PD
                init = "PD"
            else:
                #This shouldn't happen
                print("Three slices with same sequence")           

    prev_mean = 0
    prev = 0 #T2 
    cnt_t2 = 0
    cnt_pd = 0
    for i in range(int(img_np.shape[0])):
        curr_slice = img_np[i,:,:]
        curr_mean = np.mean(curr_slice)
        if i==0:
            if init=="T2":
                new_images[0,cnt_t2,:,:] = curr_slice
                prev_mean = curr_mean
                prev = 0
                if(cnt_t2 < int(img_np.shape[0]/2)-1):
                    cnt_t2 += 1
            else:
                new_images[1,cnt_pd,:,:] = curr_slice
                prev_mean = curr_mean
                prev = 1
                if(cnt_pd < int(img_np.shape[0]/2)-1):
                    cnt_pd += 1
        else:
            if np.abs(curr_mean - prev_mean) > th: #Change
                if curr_mean>prev_mean: #PD
                    new_images[1, cnt_pd, :, :] = curr_slice
                    prev_mean = curr_mean
                    prev = 1
                    if(cnt_pd < int(img_np.shape[0]/2)-1):
                        cnt_pd += 1
                else: #T2
                    new_images[0, cnt_t2, :, :] = curr_slice
                    prev_mean = curr_mean
                    prev = 0
                    if(cnt_t2 < int(img_np.shape[0]/2)-1):
                        cnt_t2 += 1
            else: # No change
                if prev==0:
                    new_images[prev, cnt_t2, :, :] = curr_slice
                    prev_mean = curr_mean
                    prev = 0 #Sobra
                    if(cnt_t2 < int(img_np.shape[0]/2)-1):
                        cnt_t2 += 1
                else:
                    new_images[prev, cnt_pd, :, :] = curr_slice
                    prev_mean = curr_mean
                    prev = 1
                    if(cnt_pd < int(img_np.shape[0]/2)-1):
                        cnt_pd += 1
            

    return new_images[0,:,:,:], new_images[1,:,:,:]

study_centers_list = list_folders(path_original)
for sc in study_centers_list[7:]:
    patients_list = list_folders(jp(path_original, sc))
    for patient in patients_list[:-1]:
        timepoints_list = list_folders(jp(path_original, sc, patient))
        for tp in timepoints_list: 
            if sc == "0010" and patient=="0003" and tp=="V3": #Exception: images corrupted
                continue

            if os.path.exists(jp(path_original, sc, patient, tp, "TSE_PD")): # If folder missing
                folder_name_dual = "TSE_PD"
            elif os.path.exists(jp(path_original, sc, patient, tp, "TSE_T2")):
                folder_name_dual = "TSE_T2"
            else:
                continue
            filename = list_files_with_extension(jp(path_original, sc, patient, tp, folder_name_dual), "nii.gz")
            if not filename: #If file missing
                continue
            img_itk = sitk.ReadImage(jp(path_original,sc,patient,tp, folder_name_dual, filename[0]))
            t2, pd = get_pd_and_t2(img_itk)
            
            t2_itk = sitk.GetImageFromArray(t2)
            pd_itk = sitk.GetImageFromArray(pd)
            #Open T1_c to copy information
            timepoints_list = list_folders(jp(path_destiny, sc, patient))
            equivalent_timepoint = find_element_in_list_that_contains(timepoints_list, tp)
            if not bool(equivalent_timepoint): # If no matching timepoint, continue.. This happens when there are more timepoints in original images as compared to processed images
                continue

            #Check if T1Gd exists. If not, continue
            if not os.path.exists(jp(path_destiny, sc, patient, equivalent_timepoint, "T1Gd", "Native")):
                continue
            list_images_t1_c = os.listdir(jp(path_destiny, sc, patient, equivalent_timepoint, "T1Gd", "Native"))
            ref_itk = sitk.ReadImage(jp(path_destiny, sc, patient, equivalent_timepoint, "T1Gd", "Native", list_images_t1_c[0]))
            if not (ref_itk.GetSize()== t2_itk.GetSize() and ref_itk.GetSize()== pd_itk.GetSize()): # If dimensions mismatch
                continue
            t2_itk.CopyInformation(ref_itk)
            pd_itk.CopyInformation(ref_itk)

            print("SC: ", sc, " Patient: ", patient, " TP: ", tp)

            #Create folders in destiny and save
            create_folder(jp(path_destiny, sc, patient, equivalent_timepoint, "T2"))
            create_folder(jp(path_destiny, sc, patient, equivalent_timepoint, "T2", "Native"))
            create_folder(jp(path_destiny, sc, patient, equivalent_timepoint, "PD"))
            create_folder(jp(path_destiny, sc, patient, equivalent_timepoint, "PD", "Native"))
            sitk.WriteImage(t2_itk, jp(path_destiny, sc, patient, equivalent_timepoint, "T2", "Native", "T2.nii.gz"))
            sitk.WriteImage(pd_itk, jp(path_destiny, sc, patient, equivalent_timepoint, "PD", "Native", "PD.nii.gz"))


