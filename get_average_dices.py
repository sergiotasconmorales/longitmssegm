import os
from os.path import join as jp
import pandas as pd
import numpy as np
from ms_segmentation.general.general import list_folders

experiment_folder = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\cross_validation\CROSS_VALIDATION_unet2dconvLSTM_2020-04-14_23_59_18'
folds = list_folders(experiment_folder)
columns = ['Case','DSC','HD']
df = pd.DataFrame(columns = columns)
for fold in folds:
    curr_df = pd.read_csv(jp(experiment_folder, fold, "results.csv"))
    df = pd.concat([df, curr_df])

dices = df['DSC'].to_numpy()
hd = df['HD'].to_numpy()
print("Mean DSC for all folds: ", np.mean(dices), "+-", np.std(dices))
print("Mean HD for all folds: ", np.mean(hd), "+-", np.std(hd))
