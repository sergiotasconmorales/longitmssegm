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

debug = False

options = {}
options['val_split']  = 0.2
#options['input_data'] = ['flair.nii.gz', 'mprage.nii.gz', 'pd.nii.gz']
options['input_data'] = ['flair.nii.gz', 'mprage.nii.gz', 'pd.nii.gz', 't2.nii.gz']
#options['input_data'] = ['flair.nii.gz', 'pd_times_flair.nii.gz', 't2_times_flair.nii.gz', 't1_inv_times_flair.nii.gz', 'sum_times_flair.nii.gz']
#options['input_data'] = ['fused_flt2.nii.gz']
options['gt'] = 'mask1.nii.gz'                     # ACHTUUUUUUUUUUUUUUUUUUNG!
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

#experiment_name = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_CS\cross_validation\CROSS_VALIDATION_UNet3D_2020-05-04_18_48_20[flair_only]'
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


    transf = transforms.Compose([   RandomFlipX(),
                                    RandomFlipY(),
                                    RandomFlipZ(),
                                    ToTensor3DPatch()
                                ])

    
    print('Training data: ')
    
    training_dataset = PatchLoader3DLoadAll(input_data=input_dictionary['input_train_data'],
                                        labels=input_dictionary['input_train_labels'],
                                        rois=input_dictionary['input_train_rois'],
                                        patch_size=options['patch_size'],
                                        sampling_step = options['sampling_step'],
                                        normalize=options['normalize'],
                                        norm_type=options['norm_type'],
                                        sampling_type=options['patch_sampling'],
                                        resample_epoch=options['resample_each_epoch'],
                                        transform=None)

    training_dataloader = DataLoader(training_dataset, 
                                    batch_size=options['batch_size'],
                                    shuffle=True)


    print('Validation data: ')
    validation_dataset = PatchLoader3DLoadAll(input_data=input_dictionary['input_val_data'],
                                            labels=input_dictionary['input_val_labels'],
                                            rois=input_dictionary['input_val_rois'],
                                            patch_size=options['patch_size'],
                                            sampling_step = options['sampling_step'],
                                            normalize=options['normalize'],
                                            norm_type=options['norm_type'],
                                            sampling_type=options['patch_sampling'],
                                            resample_epoch=options['resample_each_epoch'],
                                            transform=None)

    validation_dataloader = DataLoader(validation_dataset, 
                                    batch_size=options['batch_size'],
                                    shuffle=True)
    

    lesion_model = UNet_3D_double_skip_hybrid(n_channels=len(options['input_data']), n_classes=options['num_classes'], bilinear = False)
    #lesion_model.cuda()
    #input_tensor = torch.rand(16, 4, 32, 32, 32).cuda()
    #pred = lesion_model(input_tensor)

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

    def save_batch(x, y):
            x_np = x.detach().cpu().numpy() # 16,3,32,32,32
            y_np = y.detach().cpu().numpy() # 16,1,32,32,32
            for i_batch in range(x_np.shape[0]):
                save_image(y_np[i_batch, 0,:,:,:], "label_batch_"+ str(i_batch) +".nii.gz")
                for i_seq in range(x_np.shape[1]):
                    save_image(x_np[i_batch, i_seq, :, :, :], "img_batch_"+str(i_batch)+ "_seq_"+str(i_seq) + ".nii.gz")



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
                    # x = [batch_size, num_modalities, patch_dim1, patch_dim2, patch_dim3]
                    # y = [batch_size, 1, patch_dim1, patch_dim2, patch_dim3]
        
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
                        print("**Epoch:** ", epoch)
                        print("Training Loss: ", loss.item(), end = ", ")
                        print("Training - Batch JSC: ", batch_jacc, end = ", ")
                        #print("Num 1s: ", len(pred_labels[pred_labels>0]))
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
                        print("Validation - Batch Acc: ", accuracy)
                        acc_val.append(accuracy)
                    else:
                        lbl = y.cpu().numpy().reshape(-1)
                        pred_labels = pred.detach().cpu().numpy().astype('float32')
                        pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
                        batch_jacc = jsc(pred_labels,lbl, average = 'binary')
                        print("**Epoch:** ", epoch)
                        print("Validation Loss: ", loss.item(), end = ", ")
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
                train_jacc,
                val_loss,
                val_jacc))
            
            
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
        plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))                  
        #plot_learning_curve(train_accs, val_accs, the_title="Accuracy plot", measure = "Accuraca", early_stopping = False, filename = jp(path_results, "acc_plot.png"))                  

    options['max_epoch_reached'] = epoch

                        

    #Plot learning curve
    if train_complete:
        plot_learning_curve(train_losses, val_losses, the_title="Learning curve", measure = "Loss (" + options["loss"] + ")", early_stopping = True, filename = jp(path_results, "loss_plot.png"))
        #plot_learning_curve(train_accs, val_accs, the_title="Accuracy plot", measure = "Accuracy", early_stopping = False, filename = jp(path_results, "acc_plot.png"))                  
        plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))
    else:
        try:
            lesion_model.load_state_dict(torch.load(jp(path_models,"checkpoint.pt")))
        except:
            raise ValueError("No model found")




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

            lesion_out = build_image(infer_patches, lesion_model, device, options['num_classes'], options)

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