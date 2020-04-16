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
from ms_segmentation.general.general import create_folder, expand_dictionary, list_folders, get_experiment_name, create_log, cls, save_image
from os.path import join as jp
from ms_segmentation.plot.plot import shim_slice, shim_overlay_slice, shim, shim_overlay, plot_learning_curve
from medpy.io import load
from ms_segmentation.data_generation.slice_manager import SlicesGroupLoaderTimeLoadAll, SlicesLoader, SlicesLoaderLoadAll, get_inference_slices, get_inference_slices_time, get_probs, undo_crop_images
from ms_segmentation.data_generation.patch_manager_3d import PatchLoader3DLoadAll, build_image, get_inference_patches, reconstruct_image
from ms_segmentation.architectures.unet_c_gru import UNet_ConvGRU_2D_alt, UNet_ConvLSTM_2D_alt, UNet_ConvLSTM_Goku, UNet_ConvLSTM_Vegeta
from ms_segmentation.architectures.unet2d import UNet2D
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adadelta, Adam
import torchvision.transforms as transforms
from ms_segmentation.general.training_helper import EarlyStopping, exp_lr_scheduler, dice_loss_2d, create_training_validation_sets
from sklearn.metrics import jaccard_score as jsc
from ms_segmentation.evaluation.metrics import compute_dices, compute_hausdorf


debug = False




options = {}
options['val_split']  = 0.2
options['input_data'] = ['flair', 'mprage', 'pd'] #No format
options['gt'] = 'mask1'
options['brain_mask'] = 'brain_mask'
options['num_classes'] = 2
options['normalize'] = True 
options['norm_type'] = 'zero_one'
options['batch_size'] = 10
options['patience'] =  20 #Patience for the early stopping
options['gpu_use'] = True
options['num_epochs'] = 200
options['optimizer'] = 'adam'
options['patch_sampling'] = 'mask' # (mask, balanced or balanced+roi or non-uniform)
options['loss'] = 'dice' # (dice, cross-entropy)
options['resample_each_epoch'] = False


path_base = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L'
path_data = jp(path_base, 'isbi_train')
options['path_data'] = path_data
path_res = jp(path_base, "cross_validation")
all_patients = list_folders(path_data)

if(debug):
    experiment_name = "dummy_experiment_CROSS_VALIDATION_UNet_2D"
else: 
    experiment_name, curr_date, curr_time = get_experiment_name(the_prefix = "CROSS_VALIDATION_unet2d")

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

    # Organize the data in a dictionary 
    input_dictionary = create_training_validation_sets(options, dataset_mode="l")
    
    input_dictionary = expand_dictionary(input_dictionary)

    transf = transforms.ToTensor()

    

    print('Training data: ')
    training_dataset = SlicesLoaderLoadAll(input_data=input_dictionary['input_train_data'],
                                    labels=input_dictionary['input_train_labels'],
                                    roi=input_dictionary['input_train_rois'],
                                    normalize=options['normalize'],
                                    norm_type=options['norm_type'])

    hola = training_dataset.__getitem__(100)



    training_dataloader = DataLoader(training_dataset, 
                                    batch_size=options['batch_size'],
                                    shuffle=True)

    print('Validation data: ')
    validation_dataset = SlicesLoaderLoadAll(input_data=input_dictionary['input_val_data'],
                                        labels=input_dictionary['input_val_labels'],
                                        roi=input_dictionary['input_val_rois'],
                                        normalize=options['normalize'],
                                        norm_type=options['norm_type'])

    validation_dataloader = DataLoader(validation_dataset, 
                                    batch_size=options['batch_size'],
                                    shuffle=True)


    


    #Get frequency of each label
    if(options['loss'] == 'cross-entropy'):
        print("Computing frequency of positive and negative voxels in patches for weighted crossentropy...")
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


    from torch.optim import Adadelta, Adam

    # Define the model 
    #lesion_model = UNet_ConvGRU_2D_alt(n_channels=len(options['input_data']), n_classes = options['num_classes'], bilinear = False)
    #lesion_model = UNet_ConvLSTM_Vegeta(n_channels=len(options['input_data']), n_classes = options['num_classes'], bilinear = False)
    #lesion_model = UNet_ConvLSTM_Goku(n_channels=len(options['input_data']), n_classes = options['num_classes'], bilinear = False)
    lesion_model = UNet2D(n_channels = len(options['input_data']), n_classes = options['num_classes'], bilinear = False)

    #lesion_model = UNet2D(n_channels = len(options['input_data']), n_classes = options['num_classes'], bilinear = False)

    # lesion_model.cuda()
    # input_tensor = torch.rand(5, 3, 4, 160, 200).cuda()
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

    # training loop
    training = True
    train_complete = False
    epoch = 1

    if not training and options['loss'] == "cross-entropy":
        weights = [1.0, 40.0]
        class_weights = torch.FloatTensor(weights).cuda()

    try:
        while training:
            cls()
            # epoch specific metrics
            train_loss = 0
            val_loss = 0
            jaccs_train = []
            jaccs_val = []
            
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
        
                    x = data.type('torch.cuda.FloatTensor').to(device)
                    y = target.type('torch.cuda.FloatTensor').to(device)
                    
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
                    # compute the accuracy
                    lbl = y.cpu().numpy().reshape(-1)
                    pred_labels = pred.detach().cpu().numpy().astype('float32')
                    pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
                    batch_jacc = jsc(pred_labels,lbl, average = 'binary')
                    print("Training Loss: ", loss.item())
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
                    x = data.type('torch.cuda.FloatTensor').to(device)
                    y = target.type('torch.cuda.FloatTensor').to(device)
                    
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


                            loss = dice_loss_2d(torch.log(torch.clamp(pred, 1E-7, 1.0)), y_one_hot)
                        elif options['loss'] == 'cross-entropy':
                            loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                            y.squeeze(dim=1).long(), weight=class_weights)
                        else:
                            raise ValueError("Unknown loss")
                        
                        
                        val_loss += loss.item()
                    
                        lbl = y.cpu().numpy().reshape(-1)
                        pred_labels = pred.detach().cpu().numpy().astype('float32')
                        pred_labels = np.argmax(pred_labels, axis=1).reshape(-1).astype(np.uint8)
                        batch_jacc = jsc(pred_labels,lbl, average = 'binary')
                        print("Validation - Loss: ", loss.item())
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

    options['max_epoch_reached'] = epoch

    #Plot learning curve
    if train_complete:
        plot_learning_curve(train_losses, val_losses, the_title="Learning curve", measure = "Loss (" + options["loss"] + ")", early_stopping = True, filename = jp(path_results, "loss_plot.png"))
        plot_learning_curve(train_jaccs, val_jaccs, the_title="Jaccard plot", measure = "Jaccard", early_stopping = False, filename = jp(path_results, "jaccard_plot.png"))
    else:
        try:
            lesion_model.load_state_dict(torch.load(jp(path_models,"checkpoint.pt")))
        except:
            raise ValueError("No model found")

    
    #Evaluate all images
    columns = ['Case','DSC','HD']
    df = pd.DataFrame(columns = columns)

    test_images = [curr_test_patient]
    i_row=0
    cnt=0
    path_test = options['test_path']
    for case in test_images:
        print(cnt+1, "/", len(test_images))
        scan_path = jp(path_test, case)


        inf_slices_all = get_inference_slices_time(  the_path=path_test,
                                                the_case=case,
                                                input_data=options['input_data'], 
                                                normalize=True, 
                                                crop = False,
                                                out_size = None,
                                                norm_type = 'zero_one')

        for t in inf_slices_all.shape[1]: # for every timepoint
            inf_slices = inf_slices_all[t]
            probs = get_probs(inf_slices, lesion_model, device, options['num_classes'], options)
            labels = np.transpose(np.argmax(probs, axis=1).astype(np.uint8), (1,2,0))

            labels_gt = nib.load(jp(path_test, case, options['gt'])).get_fdata().astype(np.uint8)  #GT  

            #DSC
            dsc = compute_dices(labels_gt.flatten(), labels.flatten())
            hd = compute_hausdorf(labels_gt, labels.astype(np.uint8))

            df.loc[i_row] = [case, dsc[0], hd[0]]
            i_row += 1
            #Save result
            img_nib = nib.Nifti1Image(labels, np.eye(4))
            create_folder(jp(path_segmentations, case))
            nib.save(img_nib, jp(path_segmentations, case, case+"_segm.nii.gz"))

            cnt += 1


    df.to_csv(jp(path_results, "results.csv"), float_format = '%.5f', index = False)
    print(df.mean())
    create_log(path_results, options)




