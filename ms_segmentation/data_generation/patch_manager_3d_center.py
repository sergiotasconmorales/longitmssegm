import os
import nibabel as nib
import numpy as np
import torch
import random 
from scipy import ndimage
from os.path import join as jp
from torch.utils.data import Dataset
from operator import add 
from cc3d import connected_components as cc
from ..general.general import list_folders, list_files_with_name_containing, get_dictionary_with_paths_cs 
from .transforms3D import RandomFlipX, RandomFlipY, RandomFlipZ, RandomRotationXY, RandomRotationXZ, RandomRotationYZ, ToTensor3DPatch



class PatchLoader3DCenter(Dataset):
    """
    Dataset class for loading MRI patches from multiple modalities. Based on script utils.py provided by Sergi Valverde. 
    Difference with respect to PatchLoader3DTime: Timepoints are sliced so that more patches can be taken
    All patches are loaded at once
    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 patch_size,
                 random_pad=(0, 0, 0),
                 sampling_type='mask',
                 normalize=False,
                 norm_type='zero_one',
                 min_sampling_th=0,
                 num_pos_samples=5000,
                 resample_epoch=False,
                 transform=None,
                 phi = 1): 

        self.input_data = input_data
        self.input_labels = labels
        self.input_rois = rois
        self.pad_or_not = True
        self.patch_size = patch_size
        self.random_pad = random_pad
        self.sampling_type = sampling_type
        self.patch_half = tuple([idx // 2 for idx in self.patch_size])
        self.normalize = normalize
        self.norm_type = norm_type
        self.min_th = min_sampling_th
        self.resample_epoch = resample_epoch
        self.transform = transform
        self.num_pos_samples = num_pos_samples
        self.prev_im_ = None
        self.prev_input_data = None
        self.prev_labels = None
        self.phi = phi
        

        #Check that number of images coincide 
        if not len(input_data) == len(labels) == len(rois):
            raise ValueError("Number of input samples, input labels and input rois does not coincide")

        self.num_modalities = len(list(input_data.values())[0][0])
        self.input_train_dim = (self.num_modalities, ) + self.patch_size
        self.input_label_dim = (1, ) + self.patch_size

        self.patch_indexes = self.generate_patch_indexes()
        self.all_patches, self.all_labels = self.load_all_patches()

    def __len__(self):
        """
        Get the legnth of the training set
        """
        return len(self.patch_indexes)

    def __getitem__(self, idx):
        """
        Get the next item. Resampling the entire dataset is considered if
        self.resample_epoch is set to True.
        """
        if idx == 0 and self.resample_epoch:
            self.patch_indexes = self.generate_patch_indexes()
            self.all_patches, self.all_labels = self.load_all_patches()
        
        patches = self.all_patches[idx, :, :, :, :]

        labels = self.all_labels[idx]

        if self.transform:
            patches = self.transform((patches))
            labels = torch.Tensor(labels)
        return patches, labels
            


    def load_all_patches(self):

        all_patches = np.zeros((len(self.patch_indexes), self.num_modalities, self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype='float32')

        all_labels = np.array([x[-1] for x in self.patch_indexes])
        
        prev_pat = None
        prev_tp = None

        for idx in range(len(self.patch_indexes)):
            print(idx+1, "/", len(self.patch_indexes))
            im_ = self.patch_indexes[idx][0] #Patient
            tp = self.patch_indexes[idx][1] #Timepoint
            center = self.patch_indexes[idx][2] #Center of the patch

            slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                    for (c_idx, p_idx, s_idx) in zip(center,
                                                    self.patch_half,
                                                    self.patch_size)]

            #output_patch = np.zeros(self.input_train_dim, dtype = 'float32') #Array to store output patches
            
            if(prev_pat != im_ or prev_tp != tp): #Load new image only if it´s another patient or another timepoint
                if self.pad_or_not:
                    s = [self.apply_padding(nib.load(
                            self.input_data[im_][tp][k]).get_data().astype('float32'))
                                    for k in range(self.num_modalities)]
                else:
                    s = [nib.load(
                            self.input_data[im_][tp][k]).get_data().astype('float32')
                                    for k in range(self.num_modalities)]

                if self.normalize:
                    s = [normalize_data(s[m], norm_type = self.norm_type) for m in range(len(s))]

                prev_pat = im_
                prev_tp = tp

            # get current patches for both training data and labels
            input_train = np.stack([s[m][tuple(slice_)]
                                    for m in range(self.num_modalities)], axis=0)
        
            # check dimensions and put zeros if necessary
            if input_train.shape != self.input_train_dim:
                print('error in patch', input_train.shape, self.input_train_dim)
                input_train = np.zeros(self.input_train_dim).astype('float32')
  

            if self.transform:
                input_train, input_label = self.transform([input_train,
                                                        input_label])

            all_patches[idx,:,:,:,:] = input_train 
            
        return all_patches, all_labels

    def apply_padding(self, input_data, mode='constant', value=0):
        """
        Apply padding to edges in order to avoid overflow

        """
        padding = tuple((idx, size-idx)
                        for idx, size in zip(self.patch_half, self.patch_size))

        padded_image = np.pad(input_data,
                              padding,
                              mode=mode,
                              constant_values=value)
        return padded_image

    

    def generate_patch_indexes(self):
        """
        Generate indexes to extract. Consider the sampling step and
        a initial random padding
        """
        training_indexes = []
        # patch_half = tuple([idx // 2 for idx in self.patch_size])

        for patient_number,timepoints_list in self.input_data.items(): # For each patient
            for tp in range(len(timepoints_list)):
                #Padding
                print(">>Analyzing patient", patient_number, ", timepoint", tp+1)
                if self.pad_or_not:
                    s = [self.apply_padding(nib.load(
                            timepoints_list[tp][k]).get_data().astype('float32')) #Take first timepoint as reference for the patches
                                for k in range(self.num_modalities)]
                    l = [self.apply_padding(nib.load(
                            self.input_labels[patient_number][tp][0]).get_data().astype('float32'))] #Take GT of last timepoint
                    r = [self.apply_padding(nib.load(
                            self.input_rois[patient_number][tp][0]).get_data().astype('float32'))] #Take last brain mask 
                #No pading
                else:
                    s = [nib.load(
                            timepoints_list[tp][k]).get_data().astype('float32')
                                for k in range(self.num_modalities)]
                    l = [nib.load(
                            self.input_labels[patient_number][tp][0]).get_data().astype('float32')]
                    r = [nib.load(
                            self.input_rois[patient_number][tp][0]).get_data().astype('float32')]                


                base_img = np.zeros_like(l[0], dtype=np.uint8)

                diff = r[0] - l[0] # brain mask excluding labels
                coords_diff = np.random.permutation(np.stack(np.where(diff>0), axis = 1))

                #analize every component of the labels
                labels_out = cc(l[0].astype(np.uint8))
                num_components = np.max(labels_out)
                counter = 0
            
                for lbl in range(1, num_components+1):
                    num_voxels_lbl = np.count_nonzero(labels_out*(labels_out==lbl))
                    coords = np.random.permutation(np.stack(np.where(labels_out==lbl), axis=1))
                    num_random = int(np.ceil(self.phi*(num_voxels_lbl/self.patch_size[0])))
                    base_img[coords[:num_random, 0], coords[:num_random, 1], coords[:num_random, 2]] = 1
                    counter += num_random

                base_img[coords_diff[:counter, 0], coords_diff[:counter, 1], coords_diff[:counter, 2]] = 1

                [x, y, z] = np.where(base_img)
                
                voxel_coords = [(x_, y_, z_) for x_, y_, z_ in zip(x, y, z)]

                training_indexes += [(patient_number, tp, tuple(v), int(l[0][v])) for v in voxel_coords]
                print(len(voxel_coords), "patches selected")
        print("Total number of patches:", len(training_indexes))


        return training_indexes

    def get_candidate_voxels(self, input_mask, label_mask, roi_mask):
        """
        Sample input mask using different techniques:
        - all: extracts all voxels > 0 from the input_mask
        - mask: extracts all roi voxels
        - balanced: same number of positive and negative voxels from
                    the input_mask as defined by the roi mask
        - balanced+roi: same number of positive and negative voxels from
                    the roi and label mask

        - hybrid sampling:
          1. Set a number of positive samples == self.pos_samples
          2. Displace randomly its x, y, z position < self.patch_half
          3. Get the same number of negative samples from the roi mask
        """

        if self.sampling_type == 'image':
            sampled_mask = input_mask > 0

        if self.sampling_type == 'all':
            sampled_mask = input_mask > 0

        if self.sampling_type == 'mask':
            sampled_mask = roi_mask > 0

        if self.sampling_type == 'balanced':
            sampled_mask = label_mask > 0
            num_positive = np.sum(label_mask > 0)
            brain_voxels = np.stack(np.where(input_mask > self.min_th), axis=1)
            for voxel in np.random.permutation(brain_voxels)[:num_positive]:
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        if self.sampling_type == 'balanced+roi':
            sampled_mask = label_mask > 0
            num_positive = np.sum(label_mask > 0)
            roi_mask[label_mask == 1] = 0
            brain_voxels = np.stack(np.where(roi_mask > 0), axis=1)
            for voxel in np.random.permutation(brain_voxels)[:num_positive]:
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        if self.sampling_type == 'hybrid':
            x, y, z = np.where(label_mask > 0)
            number_of_samples = len(x)

            # sample voxels randomly until size equals self.num_samples
            if number_of_samples < self.num_pos_samples:
                expand_interval = int(self.num_pos_samples / number_of_samples) + 1
                x = np.repeat(x, expand_interval)
                y = np.repeat(y, expand_interval)
                z = np.repeat(z, expand_interval)

            index_perm = np.random.permutation(range(len(x)))
            x = x[index_perm][:self.num_pos_samples]
            y = y[index_perm][:self.num_pos_samples]
            z = z[index_perm][:self.num_pos_samples]

            # randomize the voxel center
            min_int_x = - self.patch_half[0] +1
            max_int_x = self.patch_half[0] -1
            min_int_y = - self.patch_half[1] +1
            max_int_y = self.patch_half[1] -1
            min_int_z = - self.patch_half[2] +1
            max_int_z = self.patch_half[2] -1
            x += np.random.randint(low=min_int_x,
                                   high=max_int_x,
                                    size=x.shape)
            y += np.random.randint(low=min_int_y,
                                   high=max_int_y,
                                   size=y.shape)
            z += np.random.randint(low=min_int_z,
                                   high=max_int_z,
                                   size=z.shape)

            # check boundaries
            x = np.maximum(self.patch_half[0], x)
            x = np.minimum(label_mask.shape[0] - self.patch_half[0], x)
            y = np.maximum(self.patch_half[1], y)
            y = np.minimum(label_mask.shape[1] - self.patch_half[1], y)
            z = np.maximum(self.patch_half[2], z)
            z = np.minimum(label_mask.shape[2] - self.patch_half[2], z)

            # assign the same number of positive and negative voxels
            sampled_mask = np.zeros_like(label_mask)

            # positive samples
            for x_v, y_v, z_v in zip(x, y, z):
                sampled_mask[x, y, z] = 1

            # negative samples
            brain_voxels = np.stack(np.where(roi_mask > 0), axis=1)
            for voxel in np.random.permutation(brain_voxels)[:self.num_pos_samples]:
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        return sampled_mask








def normalize_data(im,
                   norm_type='zero_one',
                   brainmask=None,
                   datatype=np.float32):
    """
    Zero mean normalization

    inputs:
    - im: input data
    - nomr_type: 'zero_one', 'standard'

    outputs:
    - normalized image
    """
    mask = np.copy(im > 0 if brainmask is None else brainmask)

    if np.count_nonzero(im) == 0:
        return im.astype(np.float32)

    if norm_type == 'standard':
        im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
        im = im / im[np.nonzero(im)].std()

    if norm_type == 'zero_one':
        min_int = abs(im.min())
        max_int = im.max()
        if im.min() < 0:
            im = im.astype(dtype=datatype) + min_int
            im = im / (max_int + min_int)
        else:
            im = (im.astype(dtype=datatype) - min_int) / max_int

    # do not apply normalization to non-brain parts
    # im[mask==0] = 0
    return im


def apply_padding(input_data, patch_size, mode='constant', value=0):
    """
    Apply padding to edges in order to avoid overflow

    """
    patch_half = tuple([idx // 2 for idx in patch_size])

    padding = tuple((idx, size-idx)
                    for idx, size in zip(patch_half, patch_size))

    padded_image = np.pad(input_data,
                            padding,
                            mode=mode,
                            constant_values=value)
    return padded_image



def get_candidate_voxels(input_mask, sel_method='all'):
    """
    Extract candidate patches.
    """

    if sel_method == 'all':
        candidate_voxels = input_mask > 0

        voxel_coords = np.stack(np.where(candidate_voxels), axis = 1)
    return candidate_voxels, voxel_coords


def get_data_channels(input_data, tp, case, patch_indexes, num_modalities, patch_size, patch_half, normalize, norm_type):

    all_patches = np.zeros((len(patch_indexes), num_modalities, patch_size[0], patch_size[1], patch_size[2]), dtype='float32')
    s = [nib.load(
            input_data[case][tp][k]).get_data().astype('float32')
                    for k in range(num_modalities)]
    if normalize:
        s = [normalize_data(s[m], norm_type = norm_type) for m in range(len(s))]
    for idx in range(len(patch_indexes)):
        #print(idx+1, "/", len(patch_indexes))
        center = patch_indexes[idx,:] #Center of the patch
        slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                for (c_idx, p_idx, s_idx) in zip(center,
                                                patch_half,
                                                patch_size)]

        #output_patch = np.zeros(self.input_train_dim, dtype = 'float32') #Array to store output patches

        # get current patches for both training data and labels
        input_train = np.stack([s[m][tuple(slice_)]
                                for m in range(num_modalities)], axis=0)

        all_patches[idx,:,:,:] = input_train 
        
    return all_patches



def get_labels(infer_patches, lesion_model, device, num_classes, options):
  sh = infer_patches.shape
  lesion_out = np.zeros((sh[0], num_classes))
  batch_size = options['batch_size']
  b =0
  # model
  lesion_model.eval()
  with torch.no_grad():
      for b in range(0, len(lesion_out), batch_size):
          x = torch.tensor(infer_patches[b:b+batch_size]).to(device)
          pred = lesion_model(x)
          # save the result back from GPU to CPU --> numpy
          lesion_out[b:b+batch_size] = pred.detach().cpu().numpy().astype('float32')
  
  labels = np.argmax(lesion_out, axis=1).astype(np.uint8)
  
  return labels


        

