import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from os import getcwd, chdir

#Functions
def list_folders(the_path):
    return [d for d in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, d))]

def filter_list_of_names(file_names):
    return [h for h in file_names if '.' not in h]

def list_files(the_path, extension):
    saved = getcwd()
    chdir(the_path)
    it = glob.glob('*.' + extension)
    chdir(saved)
    return it

def get_spacing(modality): # Same spacing for all because all of them are registered to T1Gd
    if(modality == '3DFLAIR'):
        return (0.9375, 0.9375, 3)
    elif(modality == '2DFLAIR'):
        return (0.9375, 0.9375, 3)
    elif(modality == '3DT1'):
        return (0.9375, 0.9375, 3)
    elif(modality == 'T1Gd'):
        return (0.9375, 0.9375, 3)
    else:
        return (0.9375, 0.9375, 3)    



#path_images = r'M:\Public\MasterThesis\Data\INS'
path_data = r'D:\dev\INSPIRATION'
#path_images = r'C:\Users\s.morales\Documents\2d'


centers_list = list_folders(path_data)

center_cnt = 1
for center in centers_list:
    print("Current center: ", center_cnt, "/", len(centers_list))
    path_images = os.path.join(path_data, center)

    #First, change format of T1Gd images for all cases and subcases
    cases_list = list_folders(path_images)

    #Generate volumes for all modalities and GTs for all cases and subcases
    pat_cnt = 1
    for case in cases_list:
        print("Patient: ", pat_cnt, "/", len(cases_list))
        subcases_list = list_folders(os.path.join(path_images, case))
        for subcase in subcases_list:
            modalities_list = list_folders(os.path.join(path_images, case, subcase))
            for modality in modalities_list:
                types_list = list_folders(os.path.join(path_images, case, subcase, modality)) #Type refers to raw images vs labels
                for type_ in types_list:

                    if modality=="FLAIR" and type_== "Native":
                        dcm_files = list_files(os.path.join(path_images, case, subcase, modality, type_), 'dcm') #List all dicom files
                        x_size = len(dcm_files)
                        ref_img = dcm_files[0] #Take first image as reference to check the dimensions of every slice
                        ref_img_itk = sitk.ReadImage(os.path.join(path_images, case, subcase, modality, type_, ref_img))
                        spacing = get_spacing(modality)
                        y_size = ref_img_itk.GetSize()[1]
                        z_size = ref_img_itk.GetSize()[0]
                        new_volume = np.zeros((x_size, y_size, z_size))
                        slice_index = 0
                        for img in dcm_files:
                            curr_slice = sitk.ReadImage(os.path.join(path_images, case, subcase, modality, type_, img))
                            new_volume[slice_index,:,:] = sitk.GetArrayFromImage(curr_slice)[0,:,:]
                            slice_index += 1
                        new_volume_itk = sitk.GetImageFromArray(new_volume)
                        new_volume_itk.SetSpacing(get_spacing(modality))
                        sitk.WriteImage(new_volume_itk, os.path.join(path_images, case, subcase, modality, type_, case+'_'+subcase+'_'+ modality+'_'+type_+'_nifti_volume.nii.gz'))
        pat_cnt += 1
    center_cnt += 1