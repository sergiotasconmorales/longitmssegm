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
from ms_segmentation.general.general import create_folder, list_folders, save_image, get_experiment_name, create_log, cls, get_dictionary_with_paths
from os.path import join as jp
from ms_segmentation.plot.plot import shim_slice, shim_overlay_slice, shim, shim_overlay, plot_learning_curve
from medpy.io import load
from ms_segmentation.data_generation.patch_manager_3d import PatchLoader3DLoadAll, build_image, get_inference_patches, reconstruct_image, RandomFlipX, RandomFlipY, RandomFlipZ, ToTensor3DPatch
from ms_segmentation.data_generation.patch_manager_3d_center import PatchLoader3DCenter, get_labels, get_data_channels, get_candidate_voxels
from ms_segmentation.architectures.unet3d import Unet_orig, UNet3D_1, UNet3D_2
from ms_segmentation.architectures.unet_c_gru import UNet_ConvGRU_3D_1, UNet_ConvGRU_3D_alt
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

path_data = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS'
path_train = jp(path_data, "train")
path_test = jp(path_data, "test")

debug = False

if(debug):
    experiment_name = "dummy_experiment_3DUNet_center"
else:
    experiment_name, curr_date, curr_time = get_experiment_name(the_prefix = "UNet3D_center")
path_results = jp(path_data, experiment_name)
create_folder(path_results)
path_models = jp(path_results, "models")
create_folder(path_models)
path_segmentations = jp(path_results, "results")
create_folder(path_segmentations)


options = {}
options['training_path'] = path_train
options['test_path'] = path_test
options['val_split']  = 0.2
options['input_data'] = ['flair.nii.gz', 'mprage.nii.gz', 'pd.nii.gz']
options['gt'] = 'mask1.nii.gz'
options['brain_mask'] = 'brain_mask.nii.gz'
options['num_classes'] = 2
options['patch_size'] = (11,11,11)
options['sampling_step'] = (15,15,15)
options['normalize'] = True 
options['norm_type'] = 'zero_one'
options['batch_size'] = 16
options['patience'] =  20 #Patience for the early stopping
options['gpu_use'] = True
options['num_epochs'] = 200
options['optimizer'] = 'adam'
options['patch_sampling'] = 'mask' # (mask, balanced or balanced+roi or non-uniform)
options['loss'] = 'categorical-cross-entropy' # (dice, cross-entropy, categorical-cross-entropy)
options['resample_each_epoch'] = False

input_dictionary = create_training_validation_sets(options, dataset_mode="cs")


transf = transforms.Compose([   RandomFlipX(),
                                RandomFlipY(),
                                RandomFlipZ(),
                                ToTensor3DPatch()
                            ])


print('Training data: ')

training_dataset = PatchLoader3DCenter(input_data=input_dictionary['input_train_data'],
                                       labels=input_dictionary['input_train_labels'],
                                       rois=input_dictionary['input_train_rois'],
                                       patch_size=options['patch_size'],
                                       normalize=options['normalize'],
                                       norm_type=options['norm_type'],
                                       sampling_type=options['patch_sampling'],
                                       resample_epoch=options['resample_each_epoch'],
                                       transform=None,
                                       phi = 2)

training_dataloader = DataLoader(training_dataset, 
                                 batch_size=options['batch_size'],
                                 shuffle=True,
                                 drop_last=True)


print('Validation data: ')
validation_dataset = PatchLoader3DCenter(input_data=input_dictionary['input_val_data'],
                                        labels=input_dictionary['input_val_labels'],
                                        rois=input_dictionary['input_val_rois'],
                                        patch_size=options['patch_size'],
                                        normalize=options['normalize'],
                                        norm_type=options['norm_type'],
                                        sampling_type=options['patch_sampling'],
                                        resample_epoch=options['resample_each_epoch'],
                                        transform=None,
                                        phi = 2)

validation_dataloader = DataLoader(validation_dataset, 
                                   batch_size=options['batch_size'],
                                   shuffle=True,
                                   drop_last= True)


lesion_model = CNN2(n_channels=len(options['input_data']), n_classes=options['num_classes'], bilinear=False)
# lesion_model.cuda()
# input_tensor = torch.rand(10, 3, 15, 15, 15).cuda()
# pred = lesion_model(input_tensor)

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

train_losses = []
val_losses = []
train_jaccs = []
val_jaccs = []
train_accs = []
val_accs = []

# training loop
training = True
train_complete = False
epoch = 1


try:
    while training:
        cls()
        # epoch specific metrics
        train_loss = 0
        val_loss = 0
        jaccs_train = []
        jaccs_val = []
        acc_train = []
        acc_val = []
        
        # -----------------------------
        # training samples
        # -----------------------------
        
        # set the model into train mode
        lesion_model.train() #Put in train mode
        for b_t, (data, target) in enumerate(training_dataloader):
                print("Training. Mini-batch ", b_t+1, "/", len(training_dataloader))
                # process batches: each batch is composed by training (x) and labels (y)
                # x = [batch_size, num_timepoints, num_modalities, patch_dim1, patch_dim2, patch_dim3]
                # y = [batch_size, num_timepoints, 1, patch_dim1, patch_dim2, patch_dim3]
    
                x = data.to(device)
                y = target.to(device)

                
                #save_batch(x, y)

                # clear gradients
                optimizer.zero_grad() #Set gradients to zero for every new batch so that no accummulation takes place
                
                # infer the current batch 
                pred = lesion_model(x)
                
                # pred = [batch_size, num_classes, patch_dim1, patch_dim2, patch_dim3]

                # compute the loss. 
                # we ignore the index=2

                if options['loss']=='dice':
                    y_one_hot = pred.data.clone()
                    y_one_hot[...] = 0
                    #y_one_hot = torch.FloatTensor(x.size(0), 2, x.size(2), x.size(3))
                    #y_one_hot = y_one_hot.to(device)
                    y_one_hot.scatter_(1,y.type(torch.LongTensor).to(device),1)
                    #y_one_hot.scatter_(1,y.unsqueeze(1).type(torch.LongTensor).to(device),1)

                    loss = dice_loss(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                elif options['loss'] == 'cross-entropy':
                    loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                        y.squeeze(dim=1).long(), weight=class_weights)
                elif options['loss'] == 'categorical-cross-entropy':
                    loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)), y.long())
                else:
                    raise ValueError("Unknown loss")
                
                train_loss += loss.item()
                
                # backward loss and next step
                loss.backward()
                optimizer.step()

                # compute the accuracy
                # compute the accuracy
                if options['loss'] == 'categorical-cross-entropy':
                    lbl = y.cpu().numpy()
                    pred_labels = pred.detach().cpu().numpy().astype('float32')
                    pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
                    accuracy = acc(lbl, pred_labels)
                    print("Loss: ", loss.item())
                    print("Training - Batch Acc: ", accuracy)
                    acc_train.append(accuracy)
                else:
                    lbl = y.cpu().numpy().reshape(-1)
                    pred_labels = pred.detach().cpu().numpy().astype('float32')
                    pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
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
                # we ignore the index=2

                # compute the loss. 
                if options['loss']=='dice':
                    y_one_hot = pred.data.clone()
                    y_one_hot[...] = 0
                    #y_one_hot = torch.FloatTensor(x.size(0), 2, x.size(2), x.size(3))
                    #y_one_hot = y_one_hot.to(device)
                    y_one_hot.scatter_(1,y.type(torch.LongTensor).to(device),1)
                    loss = dice_loss(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                elif options['loss'] == 'cross-entropy':
                    loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                    y.squeeze(dim=1).long(), weight=class_weights)
                elif options['loss'] == 'categorical-cross-entropy':
                    loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)), y.long())
                else:
                    raise ValueError("Unknown loss")
                
                
                val_loss += loss.item()
            
                if options['loss'] == 'categorical-cross-entropy':
                    lbl = y.cpu().numpy()
                    pred_labels = pred.detach().cpu().numpy().astype('float32')
                    pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
                    accuracy = acc(lbl, pred_labels)
                    print("Loss: ", loss.item())
                    print("Training - Batch Acc: ", accuracy)
                    acc_val.append(accuracy)
                else:
                    lbl = y.cpu().numpy().reshape(-1)
                    pred_labels = pred.detach().cpu().numpy().astype('float32')
                    pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
                    batch_jacc = jsc(pred_labels,lbl, average = 'binary')
                    print("Validation - Batch JSC: ", batch_jacc)
                    jaccs_val.append(batch_jacc)
        
        # compute mean metrics
        train_loss /= (b_t + 1)
        val_loss /= (b_v + 1)


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if options['loss'] == 'categorical-cross-entropy':
            train_acc = np.mean(np.array(acc_train))
            val_acc = np.mean(np.array(acc_val))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
        else:
            train_jacc = np.mean(np.array(jaccs_train))
            val_jacc = np.mean(np.array(jaccs_val))
            train_jaccs.append(train_jacc)
            val_jaccs.append(val_jacc)
        print('Epoch {:d} train_loss {:.4f} train_acc {:.4f} val_loss {:.4f} val_acc {:.4f}'.format(
            epoch, 
            train_loss, 
            train_acc,
            val_loss,
            val_acc))
        
        
        #Check conditions for early stopping. Save model if improvement in validation loss with respect to previous epoch
        early_stopping(val_loss, lesion_model, path_models)
        if early_stopping.early_stop:
          print("Early stopping")
          train_complete = True
          break
        
        # update epochs
        epoch += 1

        if epoch > options['num_epochs']:
            train_complete = True
            training = False
            

        # Load latest best model
    lesion_model.load_state_dict(torch.load(jp(path_models, "checkpoint.pt")))
except KeyboardInterrupt:
    # If training is stopped, load last model
    print("Training was stopped, loading last model...")
    lesion_model.load_state_dict(torch.load(jp(path_models, "checkpoint.pt")))

    plot_learning_curve(train_losses, val_losses, the_title="Learning curve", measure = "Loss (" + options["loss"] + ")", early_stopping = True, filename = jp(path_results, "loss_plot.png"))
    #plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))                  
    plot_learning_curve(train_accs, val_accs, the_title="Accuracy plot", measure = "Accuraca", early_stopping = False, filename = jp(path_results, "acc_plot.png"))                  

options['max_epoch_reached'] = epoch

                     

#Plot learning curve
if train_complete:
    plot_learning_curve(train_losses, val_losses, the_title="Learning curve", measure = "Loss (" + options["loss"] + ")", early_stopping = True, filename = jp(path_results, "loss_plot.png"))
    plot_learning_curve(train_accs, val_accs, the_title="Accuracy plot", measure = "Accuracy", early_stopping = False, filename = jp(path_results, "acc_plot.png"))                  
    #plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))
else:
    try:
        lesion_model.load_state_dict(torch.load(jp(path_models,"checkpoint.pt")))
    except:
        raise ValueError("No model found")




#Evaluate all test images
columns = columns = compute_metrics(None, None, labels_only=True)
df = pd.DataFrame(columns = columns)

test_images = list_folders(path_test)
i_row=0
cnt=0
for case in test_images:
    print(cnt+1, "/", len(test_images))
    scan_path = jp(path_test, case)

    list_images = get_dictionary_with_paths_cs([case], path_test, options['input_data']) 
    list_rois = get_dictionary_with_paths_cs([case], path_test, [options['brain_mask']])
    patch_half = tuple([idx // 2 for idx in options['patch_size']])


    timepoints = list_folders(scan_path)
    for tp in range(len(timepoints)):
        print("Timepoint ", tp+1)
        scan_path = jp(path_test, case, str(tp+1).zfill(2))
        # get candidate voxels
        mask_image = nib.load(os.path.join(scan_path, list_rois[case][tp][0])).get_fdata()

        output_segm = np.zeros_like(mask_image, dtype = np.uint8)
        _, ref_voxels = get_candidate_voxels(   mask_image,
                                            sel_method='all')

        #Divide number of voxels to be classified into 1000 groups
        division = np.linspace(0, len(ref_voxels), 1000, dtype = int)

        for i in range(len(division)-1): #Divide total number of patches into 1000 groups
            curr_ref_voxels = ref_voxels[division[i]:division[i+1],:]
            print("Group ", i, "/", len(division))
            # input images stacked as channels
            test_patches = get_data_channels(   list_images,
                                                tp,
                                                case,
                                                curr_ref_voxels, 
                                                len(options['input_data']),
                                                options['patch_size'],
                                                patch_half,
                                                normalize = options['normalize'],
                                                norm_type = options['norm_type'])

            batch_size = options['batch_size']


            output_labels_tile = get_labels(test_patches, lesion_model, device, options['num_classes'], options)
            
            #Fill output_segm with lesion_out labels in curr_ref_voxels locations
            output_segm[curr_ref_voxels[:,0], curr_ref_voxels[:,1], curr_ref_voxels[:,2]] = output_labels_tile

        

        #Compute metrics
        list_gt = get_dictionary_with_paths_cs([case], path_test, [options["gt"]])[case]

        labels_gt = nib.load(list_gt[tp][0]).get_fdata().astype(np.uint8)  #GT  

        #DSC
        metrics = compute_metrics(labels_gt, output_segm)

        df.loc[i_row] = list(metrics.values())
        i_row += 1
        #Save result
        img_nib = nib.Nifti1Image(output_segm, np.eye(4))
        create_folder(jp(path_segmentations, case))
        nib.save(img_nib, jp(path_segmentations, case, case+"_"+ str(tp+1).zfill(2) +"_segm.nii.gz"))

        cnt += 1


df.to_csv(jp(path_results, "results.csv"), float_format = '%.5f', index = False)
print(df.mean())
create_log(path_results, options)