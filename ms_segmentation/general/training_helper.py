# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script that defines classes and functions for training procedure such as early stopping and loss functions
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      Script modified from code provided by Dr. Sergi Valverde during a seminar that took place in the UdG in 2019
#
# --------------------------------------------------------------------------------------------------------------------
import os
import torch
import random
import numpy as np
from os.path import join as jp
import torch.nn.functional as F
from .general import list_files_with_name_containing, filter_list, get_dictionary_with_paths, get_dictionary_with_paths_cs


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path_experiment):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_experiment)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_experiment)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path_experiment):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), jp(path_experiment, 'checkpoint.pt'))
        self.val_loss_min = val_loss


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def create_training_validation_sets(options, dataset_mode="cs"):
    """
    Generate the input dictionaries for training and validation
    Parameters
    ----------
    options : Define the paths for the training and validation sets to be used
    must contain the following:
                options['training_path']
                options['test_path']
                options['val_split']

    dataset_mode : String to define whether the dataset is cross-sectional (cs) or longitudinal (l)

    Returns
    -------
    input_dictionary : Contains all the paths of files to feed the network.
                input_dictionary['input_train_data']
                input_dictionary['input_train_labels']....

    """
    if 'training_path' in options: #If no cross-validation paths are given
        training_scans = os.listdir(options['training_path'])
    else: #If cross-validation, list of folders is given instead of path
        training_scans = options['training_samples']
        options['training_path'] = options['path_data']

    random.shuffle(training_scans)

    if 'test_path' in options:
        test_scans = os.listdir(options['test_path']) #Test images
    else: #If cross-validation, list of folders is given instead of path
        test_scans = options['test_samples']
        options['test_path'] = options['path_data']

    t_d = int(len(training_scans) * (1 - options['val_split']))
    training_data = training_scans[:t_d] #Training images
    validation_data = training_scans[t_d:] #Validation images

    # add condition to modify distribution if 02 is validation when 01 is test or if 01 is validation when 02 is test
    if len(validation_data) == 1 and len(test_scans)==1:
        if (test_scans[0] == '01' and validation_data[0] == '02') or (test_scans[0] == '02' and validation_data[0] == '01'):
            pass
            temp = training_data.copy()
            temp = training_data.copy() # for some reason I have to do it twice or temp is not created
            temp[0] = validation_data[0] #Exchange validation patient with first training patient
            validation_data[0] = training_data[0]
            training_data = temp          



    input_dictionary = {}

    if dataset_mode == "cs":

        input_dictionary['input_train_data'] = get_dictionary_with_paths_cs(training_data, options['training_path'],options['input_data'])
        input_dictionary['input_train_labels'] = get_dictionary_with_paths_cs(training_data, options['training_path'], [options['gt']])
        input_dictionary['input_train_rois'] = get_dictionary_with_paths_cs(training_data, options['training_path'], [options['brain_mask']])

        input_dictionary['input_val_data'] = get_dictionary_with_paths_cs(validation_data, options['training_path'],options['input_data'])
        input_dictionary['input_val_labels'] = get_dictionary_with_paths_cs(validation_data, options['training_path'], [options['gt']])
        input_dictionary['input_val_rois'] = get_dictionary_with_paths_cs(validation_data, options['training_path'], [options['brain_mask']])

        input_dictionary['input_test_data'] = get_dictionary_with_paths_cs(test_scans, options['test_path'],options['input_data'])
        input_dictionary['input_test_labels'] = get_dictionary_with_paths_cs(test_scans, options['test_path'], [options['gt']])
        input_dictionary['input_test_rois'] = get_dictionary_with_paths_cs(test_scans, options['test_path'], [options['brain_mask']])

    else: #If longitudinal data, load several images for each case
        #Training
        input_dictionary['input_train_data'] = get_dictionary_with_paths(training_data, options['training_path'], options['input_data'])    
        input_dictionary['input_train_labels'] = get_dictionary_with_paths(training_data, options['training_path'], options['gt'])
        input_dictionary['input_train_rois'] = get_dictionary_with_paths(training_data, options['training_path'], options['brain_mask'])
        #Validation
        input_dictionary['input_val_data'] = get_dictionary_with_paths(validation_data, options['training_path'], options['input_data'])    
        input_dictionary['input_val_labels'] = get_dictionary_with_paths(validation_data, options['training_path'], options['gt'])
        input_dictionary['input_val_rois'] = get_dictionary_with_paths(validation_data, options['training_path'], options['brain_mask'])
        #Test
        input_dictionary['input_test_data'] = get_dictionary_with_paths(test_scans, options['test_path'], options['input_data'])    
        input_dictionary['input_test_labels'] = get_dictionary_with_paths(test_scans, options['test_path'], options['gt'])
        input_dictionary['input_test_rois'] = get_dictionary_with_paths(test_scans, options['test_path'], options['brain_mask'])
        

    return input_dictionary

## ----------------------------------------------------------------------------------------------------------------------
# Loss functions
## ----------------------------------------------------------------------------------------------------------------------


def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input.
    Modified from https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # uniques=np.unique(target.numpy())
    # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input, dim=1)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=4)#b,c,h
    num=torch.sum(num,dim=3)
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=4)#b,c,h
    den1=torch.sum(den1,dim=3)
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=4)#b,c,h
    den2=torch.sum(den2,dim=3)#b,c
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

def dice_loss_2d(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input.
    Modified from https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 5D Tensor."
    # uniques=np.unique(target.numpy())
    # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input, dim=1)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

def dice_loss_bak(output, target):
    """Compute dice among **positive** labels to avoid unbalance.
        Modified from https://github.com/DmitryUlyanov/recognition/blob/master/contrib/criterions/dice.py
    Arguments:
        output: [batch_size * height * width,  2 ] (torch.cuda.FloatTensor)
        target: [batch_size * height * width, (1)] (torch.cuda.FloatTensor)
    Returns:
        tuple contains:
        + dice loss: for back-propagation
        + accuracy: (predtrue - true) / true
        + dice overlap:  2 * predtrue * true / (predtrue - true) * 100
        + predtrue: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
        + true: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
    """

    predict = (output.max(1)[1]).float()  # {0, 1}. 0 for the original output, 1 for the binary mask
    target = (target.squeeze(1)).float()

    # Loss
    intersection = torch.sum(predict * target, 0)
    union = torch.sum(predict * target, 0) + torch.sum(target * target, 0)
    dice = 2.0 * intersection / (union + 1e-7)
    loss = 1 - dice

    # Overlap
    predtrue = predict.eq(1).float().data    # FloatTensor 0.0 / 1.0
    true = target.data                  # FloatTensor 0.0 / 1.0
    overlap = 2 * (predtrue * true).sum() / (predtrue.sum() + true.sum() + 1e-7) * 100

    # Accuracy
    acc = predtrue.eq(true).float().mean()
    #return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7), acc, overlap, predtrue, true
    return loss, acc, overlap


def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    From: https://www.jeremyjordan.me/semantic-segmentation/
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch


def tversky_loss2D(input, target, alpha, beta):
    """Computes the Tversky loss [1].

    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor [B, C, H, W]"

    probs=F.softmax(input, dim=1)
    num=probs*target#b,c,h,w--p*g       
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2) # b, c

    den1 = probs * (1 - target) #FPs
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1,dim=2)
    
    den2 = (1 - probs) * target
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)

    tversky = num/(num+alpha*den1 + beta*den2)
    tversky_eso = tversky[:,1:]
    tversky_total = -1*torch.sum(tversky_eso)/tversky_eso.size(0) # Divide by batch size

    return tversky_total


def tversky_loss3D(input, target, alpha, beta):
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor [B, C, H, W, D]"

    probs=F.softmax(input, dim=1)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=4)#b,c,h,w
    num=torch.sum(num,dim=3) # b,c,h
    num=torch.sum(num,dim=2) # b,c

    den1 = probs * (1 - target) #FPs
    den1 = torch.sum(den1, dim=4)
    den1 = torch.sum(den1,dim=3)
    den1 = torch.sum(den1,dim=2)
    
    den2 = (1 - probs) * target
    den2 = torch.sum(den2, dim=4)
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)

    tversky = num/(num+alpha*den1 + beta*den2)
    tversky_eso = tversky[:,1:]
    tversky_total = -1*torch.sum(tversky_eso)/tversky_eso.size(0) # Divide by batch size

    return tversky_total
    return 0