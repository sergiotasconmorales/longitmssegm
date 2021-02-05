import os
from os.path import join as jp
import pandas as pd
import numpy as np
from copy import deepcopy as dc
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from ms_segmentation.general.general import create_folder, list_folders, list_files_with_name_containing, list_files_with_extension
from ms_segmentation.evaluation.metrics import compute_metrics

sns.set(style="whitegrid")

path_cs = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation'
path_l = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\cross_validation'
path_gt = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\isbi_train'
path_results = r'D:\dev\ms_data\Challenges\ISBI2015\boxplots'
create_folder(path_results)

exp_cs = ['CROSS_VALIDATION_UNet3D_2020-06-04_14_54_25[all_modalities_new]', 'CROSS_VALIDATION_UNet3D_2020-06-25_07_07_15[chi-square_norm_train]']
exp_l = ['CROSS_VALIDATION_UNetConvLSTM3D_2020-06-23_21_31_39[longitudinal_chisquare_normalization_new]', 'CROSS_VALIDATION_UNetConvLSTM3D_2020-06-13_08_43_21[all_modalities_no_hist_matching_new]']
metrics = ["DSC", "LFPR", "LTPR"]


cases = {"CS (min-max norm.)": jp(path_cs, exp_cs[0]), "CS (proposed norm.)": jp(path_cs, exp_cs[0]), "L (min-max norm.)": jp(path_l, exp_l[0]), "L (proposed norm.)": jp(path_l, exp_l[1])}

col_names = compute_metrics(None, None, labels_only=True)
#total_df = pd.DataFrame(columns = col_names)

for i_case, (c_name, c_path) in enumerate(cases.items()):
    df = pd.read_csv(jp(c_path, "all_results_postprocessed_mask1.csv"))[:-1]
    built = dc(df)
    for i_m, m in enumerate(metrics):
        if i_m == 0:
            built["metric"] = m
            built["metric value"] = df[m]
        else:
            temp_df = dc(df)
            temp_df["metric"] = m
            temp_df["metric value"] = temp_df[m]
            built = built.append(temp_df)
    built["case"] = c_name
    if i_case == 0:
        total_df = dc(built)
    else:
        total_df = total_df.append(built)


ax = sns.boxplot(x = "case", y = "metric value", hue = "metric", data=total_df, palette="Set3")
a = 234