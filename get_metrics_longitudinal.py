import os
from os.path import join as jp
import pandas as pd
import numpy as np
import nibabel as nib
import cc3d
from ms_segmentation.general.general import list_folders, list_files_with_name_containing
from ms_segmentation.evaluation.metrics import compute_metrics

last_only = True

exp_folders = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\cross_validation'
gt_folder = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\histogram_matched'

gt_list = ['mask1', 'mask2']

if last_only:
    all_experiments = list_folders(exp_folders)[-1:]
else:
    all_experiments = list_folders(exp_folders)
for i_exp, exp in enumerate(all_experiments):
    print("Curr. exp: ", exp, "(", i_exp+1, "/", len(all_experiments), ")")
    for gt_curr in gt_list:
        experiment_folder = jp(exp_folders, exp)
        #gt_folder = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\isbi_train'
        gt_name = gt_curr
        patients = list_folders(gt_folder)
        post_processing = True
        post_processing_type = 'remove_small'
        min_area = 3
        folds = list_folders(experiment_folder)

        labels_for_df = compute_metrics(None, None, labels_only = True)
        global_df = pd.DataFrame(columns = labels_for_df)
        cnt_global = 0
        for gt_patient in patients: #For every patient
            print("Current patient: ", gt_patient)
            patient_df = pd.DataFrame(columns = labels_for_df)
            gt_timepoints = list_files_with_name_containing(jp(gt_folder, gt_patient), gt_name, "nii.gz")
            cnt_patient = 0
            for i_timepoint in range(len(gt_timepoints)):
                print("Current timepoint: ", i_timepoint+1, "/", len(gt_timepoints))
                curr_gt = gt_timepoints[i_timepoint]
                curr_gt_img = nib.load(curr_gt).get_fdata().astype(np.uint8)
                curr_pred = jp(experiment_folder, "fold"+ gt_patient, "results", gt_patient, gt_patient + "_" + str(i_timepoint+1).zfill(2) + "_segm.nii.gz")
                curr_pred_img = nib.load(curr_pred).get_fdata().astype(np.uint8)
                if post_processing:
                    if post_processing_type=='remove_small':
                        labels_out = cc3d.connected_components(curr_pred_img)
                        for i_cc in np.unique(labels_out):
                            if len(labels_out[labels_out == i_cc]) < min_area:
                                curr_pred_img[labels_out == i_cc] = 0
                    else:
                        raise ValueError('Unknown post-processing type')
                metrics = compute_metrics(curr_gt_img, curr_pred_img) #Dictionary with all metrics
                global_df.loc[cnt_global] = list(metrics.values())
                patient_df.loc[cnt_patient] =  list(metrics.values())
                cnt_global += 1
                cnt_patient += 1
            #Compute averages
            patient_df.loc[cnt_patient] = list(patient_df.mean())
            if post_processing:
                timepoint_filename = "results_postprocessed_"+gt_name +".csv"
            else:
                timepoint_filename = "results_" + gt_name + ".csv"
            patient_df.to_csv(jp(experiment_folder, "fold"+gt_patient, timepoint_filename), float_format = '%.5f', index = False)

        global_df.loc[cnt_global] = list(global_df.mean())
        if (post_processing):
            file_name = "all_results_postprocessed_"+gt_name+".csv"
        else:
            file_name = "all_results_"+gt_name+".csv"
        global_df.to_csv(jp(experiment_folder, file_name),float_format = '%.5f', index = False)

