# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script with the baseline model (UNet) for the segmentation of MS lesions in INSPIRATION dataset (cross-sectional version)
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
from ms_segmentation.general.general import create_folder, list_folders, get_experiment_name, create_log
from os.path import join as jp
from ms_segmentation.plot.plot import shim_slice, shim_overlay_slice, shim, shim_overlay, plot_learning_curve
from medpy.io import load
from ms_segmentation.data_generation.patch_manager_2d_new import PatchLoader2D, build_image, get_inference_patches, reconstruct_image
from ms_segmentation.architectures.unet3d import Unet3D, Unet_orig
from ms_segmentation.architectures.unet2d import UNet2D, UNet2D_s
from ms_segmentation.architectures.resnetunet import ResNetUNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import Adadelta, Adam
from ms_segmentation.general.training_helper import EarlyStopping, exp_lr_scheduler, dice_loss, dice_loss_2d, create_training_validation_sets
from sklearn.metrics import jaccard_score as jsc
from ms_segmentation.evaluation.metrics import compute_dices, compute_hausdorf


# Define paths
path_data = r'D:\dev\INS_Test'
#path_data = r'D:\dev\INS1'
path_train = jp(path_data, "train")
path_test = jp(path_data, "test")


experiment_name, curr_date, curr_time = get_experiment_name(the_prefix = "unet2d")
# experiment_name = "unet2d_2020-03-19_19_02_16"
path_results = jp(path_data, experiment_name)
create_folder(path_results)
path_models = jp(path_results, "models")
create_folder(path_models)
path_segmentations = jp(path_results, "results")
create_folder(path_segmentations)

# Visualize one subject
#img, img_header = load(jp(path_train, '00001', 'FLAIR_masked.nii.gz'))
#mask, mask_header = load(jp(path_train, '00001', 'mask_FLAIR.nii.gz'))
#shim_overlay_slice(img, mask, 25, alpha=0.4)
#shim(img, 16)

# Define options for training
options = {}
# Experiment name
options['experiment_name'] = experiment_name
# training data path
options['training_path'] = path_train
# testing data path 
options['test_path'] = path_test
# train/validation split percentage
options['val_split']  = 0.2
# Define modalities to be considered
options['input_data'] = ['FLAIR_masked.nii.gz', 'T1_masked.nii.gz']
# Number of modalities
options['num_modalities'] = len(options['input_data'])
# Define ground truth name
options['gt'] = 'mask_FLAIR.nii.gz'
# Name of brain mask images
options['brain_mask'] = 'T1_bet_mask.nii.gz'
# Num classes
options['num_classes'] = 2
# Patch size
options['patch_size'] = (32, 32)
# Sampling step
options['sampling_step'] = (16, 16)
# Normalize patches or not
options['normalize'] = True
# Normalization (scaling) type: standard or zero_one
options['norm_type'] = 'zero_one'
# Batch size 
options['batch_size'] = 15
#  Patientce for early stopping
options['patience'] =  10 
# Use GPU or not
options['gpu_use'] = True
# Number of epochs to train
options['num_epochs'] = 10
# Optimizer
options['optimizer'] = 'adam'
# Patch sampling type
options['patch_sampling'] = 'mask' # (mask, balanced or balanced+roi or non-uniform)
# Loss
options['loss'] = 'dice' # (dice, cross-entropy)
# Whether or not to re-sample each epoch for training patches
options['resample_each_epoch'] = False


# Organize the data in a dictionary -> split into training, validation and test
input_dictionary = create_training_validation_sets(options)


#Show one subject with brain mask
#case_to_show = list(input_dictionary['input_val_data'].keys())[0] #First case of validation set
#shim_overlay(nib.load(input_dictionary['input_val_data'][case_to_show][0]).get_fdata(), nib.load(input_dictionary['input_val_rois'][case_to_show][0]).get_fdata(), 16, alpha=0.5)


# Create training, validation and test patches

transf = transforms.ToTensor()



print('Training data: ')
training_dataset = PatchLoader2D(   input_data=input_dictionary['input_train_data'],
                                    labels=input_dictionary['input_train_labels'],
                                    rois=input_dictionary['input_train_rois'],
                                    patch_size=options['patch_size'],
                                    sampling_step=options['sampling_step'],
                                    normalize=options['normalize'],
                                    norm_type=options['norm_type'],
                                    sampling_type=options['patch_sampling'],
                                    resample_epoch=options['resample_each_epoch'])

f = training_dataset.__getitem__(234)

training_dataloader = DataLoader(training_dataset, 
                                 batch_size=options['batch_size'],
                                 shuffle=True)

print('Validation data: ')
validation_dataset = PatchLoader2D( input_data=input_dictionary['input_val_data'],
                                    labels=input_dictionary['input_val_labels'],
                                    rois=input_dictionary['input_val_rois'],
                                    patch_size=options['patch_size'],
                                    sampling_step=options['sampling_step'],
                                    normalize=options['normalize'],
                                    norm_type=options['norm_type'],
                                    sampling_type=options['patch_sampling'])

validation_dataloader = DataLoader(validation_dataset, 
                                   batch_size=options['batch_size'],
                                   shuffle=True)


#Get frequency of each label
if(options['loss'] == 'cross-entropy'):
    print("Counting positive and negative voxels for cross-entropy...")
    num0 = 0
    num1 = 0
    for i in range(len(training_dataset)):
        batchsito = training_dataset.__getitem__(i)[1]
        num0 += len(batchsito[batchsito==0])
        num1 += len(batchsito[batchsito==1])

    for i in range(len(validation_dataset)):
        batchsito = validation_dataset.__getitem__(i)[1]
        num0 += len(batchsito[batchsito==0])
        num1 += len(batchsito[batchsito==1])

    k = 1/(1/num0 + 1/num1)
    weights = [k/num0, k/num1]
    print("Weights for cross-entropy: ", weights)

    class_weights = torch.FloatTensor(weights).cuda()



# Train procedure

from torch.optim import Adadelta, Adam


# Define the Unet model 
#lesion_model = Unet3D(input_size=len(options['input_data']), output_size=2)
lesion_model = UNet2D(n_channels=options['num_modalities'], n_classes=options['num_classes'])
# lesion_model = UNet2D_s(n_channels=options['num_modalities'], n_classes=options['num_classes'], bilinear = False)
options['model_name'] = lesion_model.__class__.__name__
model_name = 'ms_lesion_segmentation'

# define the torch.device
device = torch.device('cuda') if options['gpu_use'] else torch.device('cpu')

# define the optimizer
if options['optimizer'] == "adam":
    optimizer = Adam(lesion_model.parameters())
elif options['optimizer'] == "adadelta":
    optimizer = Adadelta(lesion_model.parameters())

# send the model to the device
lesion_model = lesion_model.to(device)

early_stopping = EarlyStopping(patience=options['patience'], verbose=True)

# Create lists for per-epoch measures
train_losses = []
val_losses = []
train_jaccs = []
val_jaccs = []

# training loop
training = True
train_complete = False

epoch = 1

if not training and options['loss'] = "cross-entropy":
    weights = [1.0, 40.0]
    class_weights = torch.FloatTensor(weights).cuda()


try:
    while training:
    
        # Metrics for batch measures
        train_loss = 0
        val_loss = 0
        jaccs_train = []
        jaccs_val = []
        
        
        # Train
        lesion_model.train() #Put in train mode
        for b_t, (data, target) in enumerate(training_dataloader):

                print("Training. Mini-batch ", b_t+1, "/", len(training_dataloader))

                # x = [32, 2, 32, 32]
                # y = [32, 1, 32, 32]
    
                x = data.to(device)
                y = target.to(device)

                # clear gradients
                optimizer.zero_grad() #Set gradients to zero for every new batch so that no accummulation takes place
                
                # infer the current batch 
                pred = lesion_model(x)
                
                # compute the loss. 

                if options['loss']=='dice':
                    y_one_hot = pred.data.clone()
                    y_one_hot[...] = 0
                    #y_one_hot = torch.FloatTensor(x.size(0), 2, x.size(2), x.size(3))
                    #y_one_hot = y_one_hot.to(device)
                    y_one_hot.scatter_(1,y.type(torch.LongTensor).to(device),1)
                    #y_one_hot.scatter_(1,y.unsqueeze(1).type(torch.LongTensor).to(device),1)

                    loss = dice_loss_2d(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                elif options['loss'] == 'cross-entropy':
                    loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                        y.squeeze(dim=1).long(), weight=class_weights)
                else:
                    raise ValueError("Unknown loss")

                train_loss += loss.item()
                
                # backward loss and next step
                loss.backward()
                optimizer.step()

                # compute the accuracy
                lbl = y.cpu().numpy().reshape(-1)
                pred_labels = pred.detach().cpu().numpy()
                pred_labels = np.argmax(pred_labels, axis=1).reshape(-1)
                batch_jacc = jsc(pred_labels,lbl, average = 'binary')
                print("Loss: ", loss.item())
                print("Training - Batch JSC: ", batch_jacc)
                print("Num 1s: ", len(pred_labels[pred_labels>0]))
                jaccs_train.append(batch_jacc)
                
                
        # -----------------------------
        # validation samples
        # -----------------------------
    
        # set the model into train mode
        lesion_model.eval() #Put in evaluation mode. Eg so that things like dropout don't occur during evaluation
        for b_v, (data, target) in enumerate(validation_dataloader):

                print("Validation. Mini-batch ", b_v+1, "/", len(validation_dataloader))

                x = data.to(device)
                y = target.to(device)
                
                # infer the current batch 
                with torch.no_grad(): #Don't consider the gradients
                    pred = lesion_model(x)
                
                    # compute the loss. 
                    if options['loss']=='dice':
                        y_one_hot = pred.data.clone()
                        y_one_hot[...] = 0
                        #y_one_hot = torch.FloatTensor(x.size(0), 2, x.size(2), x.size(3))
                        #y_one_hot = y_one_hot.to(device)
                        y_one_hot.scatter_(1,y.type(torch.LongTensor).to(device),1)


                        loss = dice_loss_2d(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                    elif options['loss'] == 'cross-entropy':
                        loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                        y.squeeze(dim=1).long(), weight=class_weights)
                    else:
                        raise ValueError("Unknown loss")

                    val_loss += loss.item()
                
                    # compute the accuracy 
                    lbl = y.cpu().numpy().reshape(-1)
                    pred_labels = pred.detach().cpu().numpy()
                    pred_labels = np.argmax(pred_labels, axis=1).reshape(-1)
                    batch_jacc = jsc(pred_labels,lbl, average = 'binary')
                    print("Validation - Batch JSC: ", batch_jacc)
                    jaccs_val.append(batch_jacc)
                
        
        # compute mean metrics
        train_loss /= (b_t + 1)
        val_loss /= (b_v + 1)
        train_jacc = np.mean(np.array(jaccs_train))
        val_jacc = np.mean(np.array(jaccs_val))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_jaccs.append(train_jacc)
        val_jaccs.append(val_jacc)

        print('Epoch {:d} train_loss {:.4f} train_jacc {:.4f} val_loss {:.4f} val_jacc {:.4f}'.format(
            epoch, 
            train_loss, 
            train_jacc,
            val_loss,
            val_jacc)) #Measures are shown for patches
        
        
        early_stopping(val_loss, lesion_model, path_models)
        if early_stopping.early_stop:
          print("Early stopping")
          break

        # update epochs
        epoch += 1

        if epoch > options['num_epochs']:
            train_complete = True
            training = False
            

    lesion_model.load_state_dict(torch.load(jp(path_models, "checkpoint.pt")))
except KeyboardInterrupt:
    # If training is stopped, load last model
    print("Training was stopped, loading last model...")
    lesion_model.load_state_dict(torch.load(jp(path_models, "checkpoint.pt")))

    plot_learning_curve(train_losses, val_losses, the_title="Learning curve", measure = "Loss (" + options["loss"] + ")", early_stopping = True, filename = jp(path_results, "loss_plot.png"))
    plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))                  

#Plot learning curve
if train_complete:
    plot_learning_curve(train_losses, val_losses, the_title="Learning curve", measure = "Loss (" + options["loss"] + ")", early_stopping = True, filename = jp(path_results, "loss_plot.png"))
    plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))
else:
    try:
        lesion_model.load_state_dict(torch.load(jp(path_models,"checkpoint.pt")))
    except:
        raise ValueError("No model found")



#Evaluate all test images and get metrics
columns = ['Case','DSC','HD']
df = pd.DataFrame(columns = columns)

test_images = list_folders(path_test)
i_row=0
cnt = 0
for case in test_images:
    print(cnt+1, "/", len(test_images))
    scan_path = jp(path_test, case)

    infer_patches, coordenates = get_inference_patches(scan_path=scan_path,
                                            input_data=options['input_data'],
                                            roi="T1_bet_mask.nii.gz",
                                            patch_shape=options['patch_size'],
                                            step=options['sampling_step'],
                                            normalize=options['normalize'],
                                            norm_type=options['norm_type'])

    batch_size = options['batch_size']

    lesion_out = build_image(infer_patches, lesion_model, device, 2, options)

    scan_numpy = nib.load(jp(scan_path, "FLAIR_masked.nii.gz")).get_fdata()
    all_probs = np.zeros((scan_numpy.shape[0], scan_numpy.shape[1], scan_numpy.shape[2], 2))

    for i in range(options['num_classes']):
        all_probs[:,:,:,i] = reconstruct_image(lesion_out[:,i], 
                                        coordenates, 
                                        scan_numpy.shape)
                                
    labels = np.argmax(all_probs, axis=3).astype(np.uint8)

    # Post processing - TODO

    #shim_overlay(scan_numpy, labels, 16, alpha=0.6)

    #Compute metrics
    labels_gt = nib.load(jp(path_test, case, "mask_FLAIR.nii.gz")).get_fdata().astype(np.uint8)  #GT  

    #DSC
    dsc = compute_dices(labels_gt.flatten(), labels.flatten())
    hd = compute_hausdorf(labels_gt, labels.astype(np.uint8))

    df.loc[i_row] = [str(case), dsc[0], hd[0]]
    i_row += 1
    #Save result
    img_nib = nib.Nifti1Image(labels, np.eye(4))
    create_folder(jp(path_segmentations, case))
    nib.save(img_nib, jp(path_segmentations, case, case+"_segm.nii.gz"))
    cnt += 1

df.to_csv(jp(path_results, "results.csv"), float_format = '%.5f', index = False)
print(df.mean())
create_log(path_results, options)
