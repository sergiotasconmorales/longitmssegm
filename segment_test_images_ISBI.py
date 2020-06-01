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
from ms_segmentation.general.general import parse_log_file, create_folder, list_folders, save_image, get_experiment_name, create_log, cls, get_dictionary_with_paths
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


#path_test_cs = r'D:\dev\ms_data\Challenges\ISBI2015\sample1'
path_test_cs = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\cross_sectional'
path_test_l = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\longitudinal'
path_experiments_cs = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation'
path_experiments_l = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\cross_validation'
#path_results = r'D:\dev\ms_data\Challenges\ISBI2015\res'
path_results = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs'
create_folder(path_results)
use_gpu = True


# Name of experiment to evaluate
experiment_name = 'CROSS_VALIDATION_UNet3D_2020-05-04_18_48_20[flair_only]'
experiment_type = 'cs' # cs or l
create_folder(jp(path_results, experiment_name))
device = torch.device('cuda') if use_gpu else torch.device('cpu')

all_indexes = {}
case_index = 0

# Process experiment

if experiment_type == 'cs': # if experiment is cross-sectional
    # cross-sectional
    path_exp = jp(path_experiments_cs, experiment_name)
    folds = list_folders(path_exp)

    #Read log file to get parameters
    parameters_dict = parse_log_file(jp(path_exp, folds[0])) # take file of first fold as reference

    all_folds = []
    
    for f in folds:
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
    print("Ensembling models of all folds...")
    all_segmentations = np.stack(all_folds) # size  (5, 61, 181,217, 181)
    the_sum = np.sum(all_segmentations, axis = 0) # size should be (61, 181,217, 181)
    results = the_sum >= 3 # boolean with size (61, 181,217, 181)
    
    # save images
    for i_case in range(results.shape[0]):
        save_image(results[i_case,:,:,:].astype(np.uint8), jp(path_results, experiment_name,"test"+all_indexes[i_case][0] + "_" + all_indexes[i_case][1] + "_qwertz.nii"))

    z = 0


elif experiment_type == 'l':
    pass
else:
    raise Exception("Unknown experiment_type!")
