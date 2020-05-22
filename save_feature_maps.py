import os
import random
import torch
import nibabel as nib
import numpy as np  
import pandas as pd
from ms_segmentation.general.general import create_folder, list_folders, save_image, get_experiment_name, create_log, cls, get_dictionary_with_paths
from os.path import join as jp
from ms_segmentation.plot.plot import shim_slice, shim_overlay_slice, shim, shim_overlay, plot_learning_curve
from medpy.io import load
from ms_segmentation.data_generation.patch_manager_3d import PatchLoader3DLoadAll, build_image, get_inference_patches, reconstruct_image, RandomFlipX, RandomFlipY, RandomFlipZ, ToTensor3DPatch
from ms_segmentation.architectures.unet3d import UNet_3D_alt
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

debug = True

options = {}
options['val_split']  = 0.2
options['input_data'] = ['flair.nii.gz', 'mprage.nii.gz', 'pd.nii.gz', 't2.nii.gz']
#options['input_data'] = ['flair.nii.gz', 'pd_times_flair.nii.gz', 't2_times_flair.nii.gz', 't1_inv_times_flair.nii.gz', 'sum_times_flair.nii.gz']
#options['input_data'] = ['flair.nii.gz']
options['gt'] = 'mask1.nii.gz'
options['brain_mask'] = 'brain_mask.nii.gz'
options['num_classes'] = 2
options['patch_size'] = (32,32,32)
options['sampling_step'] = (16,16,16)
options['normalize'] = True 
options['norm_type'] = 'zero_one'
options['batch_size'] = 16
options['patience'] =  20 #Patience for the early stopping
options['gpu_use'] = True
options['num_epochs'] = 200
options['optimizer'] = 'adam'
options['patch_sampling'] = 'mask' # (mask, balanced or balanced+roi or non-uniform)
options['loss'] = 'dice' # (dice, cross-entropy, categorical-cross-entropy)
options['resample_each_epoch'] = False


path_base = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS'
path_data = jp(path_base, 'isbi_cs')
options['path_data'] = path_data
path_res = jp(path_base, "cross_validation")
all_patients = list_folders(path_data)


if(debug):
    experiment_name = "dummy_experiment_CV_3DUNet"
else:
    experiment_name, curr_date, curr_time = get_experiment_name(the_prefix = "CROSS_VALIDATION_UNet3D")


# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

experiment_name = 'CROSS_VALIDATION_UNet3D_2020-05-16_10_35_36[all_modalities]'
fold = 0
for curr_test_patient in all_patients:
    fold += 1
    curr_train_patients = all_patients.copy()
    curr_train_patients.remove(curr_test_patient)

    options['training_samples'] = curr_train_patients
    options['test_samples'] = [curr_test_patient]

    experiment_folder = jp(path_res, experiment_name) 
    create_folder(experiment_folder)

    path_results = jp(experiment_folder, "fold" + str(fold).zfill(2))
    create_folder(path_results)
    path_models = jp(path_results, "models")
    create_folder(path_models)
    path_segmentations = jp(path_results, "results")
    create_folder(path_segmentations)


    input_dictionary = create_training_validation_sets(options, dataset_mode="cs")


    lesion_model = UNet_3D_alt(n_channels=len(options['input_data']), n_classes=options['num_classes'], bilinear = False)



    options['model_name'] = lesion_model.__class__.__name__
    model_name = 'ms_lesion_segmentation'


    # define the torch.device
    device = torch.device('cuda') if options['gpu_use'] else torch.device('cpu')

    lesion_model = lesion_model.to(device)

    lesion_model.load_state_dict(torch.load(jp(path_models,"checkpoint.pt")))



    #Evaluate all test images
    columns = columns = compute_metrics(None, None, labels_only=True)
    df = pd.DataFrame(columns = columns)

    test_images = [curr_test_patient]
    i_row=0
    cnt=0
    path_test = options['test_path']
    for case in test_images:
        print(cnt+1, "/", len(test_images))
        path_timepoints = jp(path_test, case)

        list_images = get_dictionary_with_paths_cs([case], path_test, options['input_data']) 
        list_rois = get_dictionary_with_paths_cs([case], path_test, [options['brain_mask']])
        patch_half = tuple([idx // 2 for idx in options['patch_size']])


        timepoints = list_folders(path_timepoints)    
        
        # get candidate voxels
        all_infer_patches, all_coordenates = get_inference_patches( path_test=path_test,
                                                            case = case,
                                                            input_data=options['input_data'],
                                                            roi=options['brain_mask'],
                                                            patch_shape=options['patch_size'],
                                                            step=options['sampling_step'],
                                                            normalize=options['normalize'],
                                                            mode = "cs"
                                                            )
            
        for tp in range(len(timepoints)):
            print("Timepoint ", tp+1)
            infer_patches = all_infer_patches[tp]
            coordenates = all_coordenates[tp]
            scan_path = jp(path_test, case, str(tp+1).zfill(2))

            lesion_out = build_image(infer_patches, lesion_model, device, options['num_classes'], options, save_feature_maps=True)

            scan_numpy = nib.load(jp(scan_path, options['brain_mask'])).get_fdata()
            all_probs = np.zeros((scan_numpy.shape[0], scan_numpy.shape[1], scan_numpy.shape[2], options['num_classes']))

            for i in range(options['num_classes']):
                all_probs[:,:,:,i] = reconstruct_image(lesion_out[:,i], 
                                                coordenates, 
                                                scan_numpy.shape)
                                    
            labels = np.argmax(all_probs, axis=3).astype(np.uint8)

            #Compute metrics
            list_gt = get_dictionary_with_paths_cs([case], path_test, [options["gt"]])[case]

            labels_gt = nib.load(list_gt[tp][0]).get_fdata().astype(np.uint8)  #GT  

            #DSC
            metrics = compute_metrics(labels_gt, labels)

            df.loc[i_row] = list(metrics.values())
            i_row += 1
            #Save result
            img_nib = nib.Nifti1Image(labels, np.eye(4))
            create_folder(jp(path_segmentations, case))
            nib.save(img_nib, jp(path_segmentations, case, case+"_"+ str(tp+1).zfill(2) +"_segm.nii.gz"))

            cnt += 1


    df.to_csv(jp(path_results, "results.csv"), float_format = '%.5f', index = False)
    print(df.mean())
    create_log(path_results, options)