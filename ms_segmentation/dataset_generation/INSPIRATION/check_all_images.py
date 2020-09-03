import os
import SimpleITK as sitk
from os.path import join as jp
import pandas as pd
from ms_segmentation.general import general as misc


path_orig = r'D:\dev\ms_data\Inspiration\INSPIRATION-org'
path_proc = r'D:\dev\ms_data\Inspiration\INSPIRATION'

# first, process original images
centers = misc.list_folders(path_orig)
columns = ['Center','Patient','Time Point', 'Modality', 'Size']
df_orig = pd.DataFrame(columns = columns)
i_orig = 0
for i_c, c in enumerate(centers): # for each study center
    patients = misc.list_folders(jp(path_orig, c)) # list of patients for center c
    for i_pat,pat in enumerate(patients): #for every patient
        timepoints = misc.list_folders(jp(path_orig, c, pat)) #list of timepoints
        for tp in timepoints:
            modalities = misc.list_folders(jp(path_orig, c, pat, tp)) # List of available modalities
            for mod in modalities:
                imgs = misc.list_files_with_extension(jp(path_orig, c, pat, tp, mod), "nii.gz") # List nii.gz images
                if len(imgs)>1: # If more than one image for a certain modality
                    df_orig.loc[i_orig] = [c, pat,  tp, mod, 'multiple']
                    i_orig += 1
                else: # If not
                    img = sitk.GetArrayFromImage(sitk.ReadImage(jp(path_orig, c, pat, tp, mod, imgs[0]))) # Read image
                    img_size = img.shape  # Get image size
                    df_orig.loc[i_orig] = [c, pat,  tp, mod, str(img_size)] # write info to df
                    #print(c, pat, tp, mod, img_size)
                    i_orig += 1
                print("Center ", i_c+1, " of ", len(centers), ", patient ", i_pat+1, " of ", len(patients))

df_orig.to_csv(jp(path_orig, 'images_info_orig.csv'))






# then, process processed images

