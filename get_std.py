import os
from os.path import join as jp
import pandas as pd
import numpy as np
import nibabel as nib
import cc3d
from ms_segmentation.general.general import list_folders, list_files_with_name_containing, list_files_with_extension
from ms_segmentation.evaluation.metrics import compute_metrics

path_experiment = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation\CROSS_VALIDATION_UNet3D_2020-06-25_07_07_15[chi-square_norm_train]'

masks =['mask1', 'mask2']

csv_files = list_files_with_extension(path_experiment, 'csv')

labels_for_df = compute_metrics(None, None, labels_only = True)
df_std = pd.DataFrame(columns = labels_for_df)
cnt = 0

for m in masks:
    f = [fi for fi in csv_files if m in fi][0]
    df = pd.read_csv(jp(path_experiment,f))[:-1] # ignore last row which is the average
    df_std.loc[cnt] = list(df.std())
    cnt += 1

df_std.to_csv(jp(path_experiment, 'std_mask1_mask2.csv'), float_format = '%.5f', index = False)