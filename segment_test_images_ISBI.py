# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script to produce segmentation masks for test images of ISBI MS lesion segmentation challenge
#               Ensemble of 5 models generated during cross-validation is done for a specific experiment and several experiments can be ensembled too
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------

import os
import random
import torch
import nibabel as nib
import numpy as np  
import pandas as pd
from ms_segmentation.general.general import get_groups, parse_log_file, create_folder, list_folders, save_image, get_experiment_name, create_log, cls, get_dictionary_with_paths, list_files_with_name_containing
from os.path import join as jp
from ms_segmentation.plot.plot import shim_slice, shim_overlay_slice, shim, shim_overlay, plot_learning_curve
from medpy.io import load
from ms_segmentation.data_generation.patch_manager_3d import PatchLoader3DLoadAll, build_image, get_inference_patches, reconstruct_image, RandomFlipX, RandomFlipY, RandomFlipZ, ToTensor3DPatch
from ms_segmentation.architectures.unet3d import UNet_3D_alt, UNet_3D_double_skip_hybrid
from ms_segmentation.architectures.unet_c_gru import UNet_ConvGRU_3D_1, UNet_ConvLSTM_3D_alt
from ms_segmentation.architectures.cnn1 import CNN1, CNN2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adadelta, Adam
import torchvision.transforms as transforms
from ms_segmentation.general.training_helper import EarlyStopping, exp_lr_scheduler, dice_loss, create_training_validation_sets, get_dictionary_with_paths_cs
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score as acc
from ms_segmentation.evaluation.metrics import compute_metrics
from torch.optim import Adadelta, Adam
import cc3d

# Name of experiment to evaluate
experiment_name = 'CROSS_VALIDATION_UNet3D_2020-05-03_18_31_37[alt_images]'
experiment_type = 'cs' # cs or l


#path_test_cs = r'D:\dev\ms_data\Challenges\ISBI2015\sample1'
path_test_cs = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\cross_sectional'
path_test_l = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\longitudinal'
path_experiments_cs = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation'
path_experiments_l = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\cross_validation'
#path_results = r'D:\dev\ms_data\Challenges\ISBI2015\res'
if experiment_type == 'cs':
    path_results = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs'
else:
    path_results = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_l'

create_folder(path_results)
use_gpu = True

def get_result_name(the_paths, the_base):
    accum = 0
    for p in the_paths:
        num_ensembles = len(list_folders(jp(p, "ensembles")))
        num_non_ensembles = len(list_folders(p)) - 1 # minus ensembles folder
        total = num_ensembles + num_non_ensembles
        accum += total
    new_name = the_base + str(accum+1).zfill(2)
    return new_name


experiment_name_folder = get_result_name([r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs', r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_l'], "qwertz")
create_folder(jp(path_results, experiment_name_folder))
device = torch.device('cuda') if use_gpu else torch.device('cpu')

all_indexes = {}
case_index = 0
post_processing = True
min_area = 3
selection = {"fold01": False, "fold02": True, "fold03": True, "fold04": False, "fold05": True}
if list(selection.values()).count(True)%2 == 0:
    raise Exception("Number of folds to consider should be odd") 



def divide_inference_slices(all_slices, desired_timepoints):
    #Format for all_slices is (num_slices, num_timepoints, num_modalities, Height, Width)
    _, num_timepoints, _, _, _, _ = all_slices.shape
    
    if(num_timepoints == desired_timepoints): #Special case if number of timepoints coincides with desired number of timepoints
        #TODO: special case -> How to predict timepoint in the middle
        pass

    list_inference_patches = []
    indexes = []
    num_forward = num_timepoints - desired_timepoints + 1
    for i in range(num_forward):
        list_inference_patches.append(all_slices[:,i:i+desired_timepoints,:,:,:,:])
        indexes.append(i+desired_timepoints)

    all_slices = np.flip(all_slices, axis = 1).copy()
    num_backward = num_forward
    if(num_timepoints%2 == 0): #even
        init = 0
    else: #odd
        init = 1
    for i in range(init,num_backward):
        list_inference_patches.append(all_slices[:,i:i+desired_timepoints,:,:,:,:])
        if(num_timepoints%2==0): #even
            indexes.append(i+1)
        else:
            indexes.append(i)
    
    sorted_indexes = np.argsort(indexes)
    sorted_list_inference_patches = [list_inference_patches[k] for k in sorted_indexes]

    return sorted_list_inference_patches

# Process experiment

if experiment_type == 'cs': # if experiment is cross-sectional
    # cross-sectional
    path_exp = jp(path_experiments_cs, experiment_name)
    folds = list_folders(path_exp)

    #Read log file to get parameters
    parameters_dict = parse_log_file(jp(path_exp, folds[0])) # take file of first fold as reference

    all_folds = []
    
    for f in folds:
        if selection[f] == False:
            continue
        fold_segmentations = []
        # create model
        lesion_model = eval(parameters_dict["model_name"])(n_channels=len(eval(parameters_dict['input_data'])), n_classes=2, bilinear = False)
        lesion_model.cuda()

        # try to load the weights
        lesion_model.load_state_dict(torch.load(jp(path_exp, f, "models","checkpoint.pt")))

        test_images = list_folders(path_test_cs) # all test cases

        cnt=0
        path_test = path_test_cs
        for case in test_images:
            path_timepoints = jp(path_test, case)

            list_images = get_dictionary_with_paths_cs([case], path_test, eval(parameters_dict['input_data'])) 
            list_rois = get_dictionary_with_paths_cs([case], path_test, [parameters_dict['brain_mask']])
            patch_half = tuple([idx // 2 for idx in eval(parameters_dict['patch_size'])])


            timepoints = list_folders(path_timepoints)    
            
            # get candidate voxels
            all_infer_patches, all_coordenates = get_inference_patches( path_test=path_test,
                                                                case = case,
                                                                input_data=eval(parameters_dict['input_data']),
                                                                roi=parameters_dict['brain_mask'],
                                                                patch_shape=eval(parameters_dict['patch_size']),
                                                                step=eval(parameters_dict['sampling_step']),
                                                                normalize=eval(parameters_dict['normalize']),
                                                                mode = "cs"
                                                                )
                
            for tp in range(len(timepoints)):
                cls()
                print("Fold: ", f)
                print("Patient", cnt+1, "/", len(test_images))
                print("Timepoint ", tp+1)
                infer_patches = all_infer_patches[tp]
                coordenates = all_coordenates[tp]
                scan_path = jp(path_test, case, str(tp+1).zfill(2))
                aux_dict = {'batch_size': eval(parameters_dict['batch_size'])}
                lesion_out = build_image(infer_patches, lesion_model, device, 2, aux_dict)

                scan_numpy = nib.load(jp(scan_path, parameters_dict['brain_mask'])).get_fdata()
                all_probs = np.zeros((scan_numpy.shape[0], scan_numpy.shape[1], scan_numpy.shape[2], 2))

                for i in range(2): #2 classes
                    all_probs[:,:,:,i] = reconstruct_image(lesion_out[:,i], 
                                                    coordenates, 
                                                    scan_numpy.shape)
                                        
                labels = np.argmax(all_probs, axis=3).astype(np.uint8)
                fold_segmentations.append(labels) # Save segmentation for every timepoint and every patient for the current fold

                #Save result
                #create_folder(jp(path_results, experiment_name, f))
                #save_image(labels, jp(path_results, experiment_name, f, str(case) + "_" + str(tp+1).zfill(2) + "_segm.nii.gz"))
                all_indexes[case_index] = [case, str(tp+1).zfill(2)]
                case_index += 1
            cnt += 1
        
        all_folds.append(np.stack(fold_segmentations))

    # ensemble segmentations of all 5 folds
    print("Ensembling models of all selected folds...")
    all_segmentations = np.stack(all_folds) # size  (5, 61, 181,217, 181)
    the_sum = np.sum(all_segmentations, axis = 0) # size should be (61, 181,217, 181)
    results = the_sum >= np.ceil(list(selection.values()).count(True)/2) # boolean with size (61, 181,217, 181)
    
    # save images
    results = results.astype(np.uint8)
    for i_case in range(results.shape[0]):
        if post_processing: #Remove very small lesions (3 voxels)
            labels_out = cc3d.connected_components(results[i_case,:,:,:])
            for i_cc in np.unique(labels_out):
                if len(labels_out[labels_out == i_cc]) < min_area:
                    results[i_case,:,:,:][labels_out == i_cc] = 0
        save_image(results[i_case,:,:,:], jp(path_results, experiment_name_folder,"test"+all_indexes[i_case][0] + "_" + all_indexes[i_case][1] + "_qwertz.nii"))

    # Save dictionary that identifies which folds were considered
    selection["experiment"] = experiment_name
    selection["Postprocessing"] = post_processing
    selection["Min-area"] = min_area
    create_log(jp(path_results, experiment_name_folder), selection)

    z = 0


elif experiment_type == 'l':
    path_exp = jp(path_experiments_l, experiment_name)
    folds = list_folders(path_exp)

    #Read log file to get parameters
    parameters_dict = parse_log_file(jp(path_exp, folds[0])) # take file of first fold as reference

    all_folds = []

    for f in folds:
        if selection[f] == False:
            continue
        fold_segmentations = []
        # create model
        parameters_dict["model_name"] = 'UNet_ConvLSTM_3D_alt'
        lesion_model = eval(parameters_dict["model_name"])(n_channels=eval(parameters_dict['num_timepoints']), n_classes=2, bilinear = False)
        lesion_model.cuda()

        # try to load the weights
        lesion_model.load_state_dict(torch.load(jp(path_exp, f, "models","checkpoint.pt")))

        test_images = list_folders(path_test_cs) # all test cases    

        cnt=0
        path_test = path_test_l
        for case in test_images:
            print(cnt+1, "/", len(test_images))
            scan_path = jp(path_test, case)

            tot_timepoints = len(list_files_with_name_containing(jp(path_test, case), "brain_mask", "nii.gz"))

            infer_patches, coordenates = get_inference_patches(path_test=path_test,
                                                    case = case,
                                                    input_data=eval(parameters_dict['input_data']),
                                                    roi=parameters_dict['brain_mask'],
                                                    patch_shape=eval(parameters_dict['patch_size']),
                                                    step=eval(parameters_dict['sampling_step']),
                                                    normalize=eval(parameters_dict['normalize']),
                                                    mode = "l",
                                                    num_timepoints=tot_timepoints)

            if 'LSTM' in parameters_dict["model_name"]:
                inf_patches_sets = divide_inference_slices(infer_patches, eval(parameters_dict['num_timepoints']))
            else:
                inf_patches_sets = get_groups(infer_patches, tot_timepoints, eval(parameters_dict['num_timepoints'])) #group patches to predict every timepoint


            batch_size = eval(parameters_dict['batch_size'])

            for i_timepoint in range(len(inf_patches_sets)):
                cls()
                print("Fold: ", f)
                print("Patient", cnt+1, "/", len(test_images))
                print("Timepoint ", i_timepoint+1)
                infer_patches = inf_patches_sets[i_timepoint]
                aux_dict = {'batch_size': eval(parameters_dict['batch_size'])}
                lesion_out = build_image(infer_patches, lesion_model, device, 2, aux_dict)

                scan_numpy = nib.load(jp(path_test, case, os.listdir(scan_path)[0])).get_fdata()
                all_probs = np.zeros((scan_numpy.shape[0], scan_numpy.shape[1], scan_numpy.shape[2], 2))

                for i in range(2):
                    all_probs[:,:,:,i] = reconstruct_image(lesion_out[:,i], 
                                                    coordenates, 
                                                    scan_numpy.shape)
                                        
                labels = np.argmax(all_probs, axis=3).astype(np.uint8)
                fold_segmentations.append(labels) # Save segmentation for every timepoint and every patient for the current fold
                #Save result
                all_indexes[case_index] = [case, str(i_timepoint+1).zfill(2)]
                case_index += 1
                #create_folder(jp(path_segmentations, case))
                #nib.save(img_nib, jp(path_segmentations, case, case+"_"+ str(i_timepoint+1).zfill(2) +"_segm.nii.gz"))

            cnt += 1
        all_folds.append(np.stack(fold_segmentations))


    # ensemble segmentations of all 5 folds
    print("Ensembling models of all selected folds...")
    all_segmentations = np.stack(all_folds) # size  (5, 61, 181,217, 181)
    the_sum = np.sum(all_segmentations, axis = 0) # size should be (61, 181,217, 181)
    results = the_sum >= np.ceil(list(selection.values()).count(True)/2) # boolean with size (61, 181,217, 181)
    
    # save images
    results = results.astype(np.uint8)
    for i_case in range(results.shape[0]):
        if post_processing: #Remove very small lesions (3 voxels)
            labels_out = cc3d.connected_components(results[i_case,:,:,:])
            for i_cc in np.unique(labels_out):
                if len(labels_out[labels_out == i_cc]) < min_area:
                    results[i_case,:,:,:][labels_out == i_cc] = 0
        save_image(results[i_case,:,:,:], jp(path_results, experiment_name_folder,"test"+all_indexes[i_case][0] + "_" + all_indexes[i_case][1] + "_qwertz.nii"))

    # Save dictionary that identifies which folds were considered
    selection["experiment"] = experiment_name
    selection["Postprocessing"] = post_processing
    selection["Min-area"] = min_area
    create_log(jp(path_results, experiment_name_folder), selection)        
else:
    raise Exception("Unknown experiment_type!")
