import os
import nibabel as nib
import numpy as np
import torch
import random 
from scipy import ndimage
from os.path import join as jp
from torch.utils.data import Dataset
from operator import add 
from ..general.general import list_folders, list_files_with_name_containing, get_dictionary_with_paths 
from .transforms3D import RandomFlipX, RandomFlipY, RandomFlipZ, RandomRotationXY, RandomRotationXZ, RandomRotationYZ, ToTensor3DPatch

class PatchLoader3D(Dataset):
    """
    Dataset class for loading MRI patches from multiple modalities. Based on script utils.py provided by Sergi Valverde. 

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 patch_size,
                 sampling_step,
                 random_pad=(0, 0, 0),
                 sampling_type='mask',
                 normalize=False,
                 norm_type='zero_one',
                 min_sampling_th=0,
                 num_pos_samples=5000,
                 resample_epoch=False,
                 transform=None):
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - patch_size: patch size
        - sampling_step: sampling_step
        - sampling type: 'all: all voxels in input_mask,
                            'roi: all voxels in roi_mask,
                            'balanced: same number of positive and negative voxels
        - normalize: Normalize data (0 mean / 1 std)
        - min_sampling_th: Minimum value to extract samples (0 default)
        - num_pos_samples used when hybrid sampling
        - transform
        """
        self.input_data = list(input_data.values())
        self.input_labels = list(labels.values())
        self.input_rois = list(rois.values())
        self.pad_or_not = True
        self.patch_size = patch_size
        self.sampling_step = sampling_step
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

        #Check that number of images coincide 
        if not len(input_data) == len(labels) == len(rois):
            raise ValueError("Number of input samples, input labels and input rois does not coincide")

        self.num_modalities = len(list(input_data.values())[0])
        self.input_train_dim = (self.num_modalities, ) + self.patch_size
        self.input_label_dim = (1, ) + self.patch_size

        self.patch_indexes = self.generate_patch_indexes()


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

        im_ = self.patch_indexes[idx][0]
        center = self.patch_indexes[idx][1]

        slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                  for (c_idx, p_idx, s_idx) in zip(center,
                                                   self.patch_half,
                                                   self.patch_size)]

        #Read images -> Super slow, reads image for every patch
        if self.pad_or_not:
            if self.prev_im_ == im_:
                s = self.prev_input_data
                l = self.prev_labels
            else:
                s = [self.apply_padding(nib.load(
                        self.input_data[im_][k]).get_data().astype('float32'))
                                for k in range(self.num_modalities)]
                l = [self.apply_padding(nib.load(
                        self.input_labels[im_][0]).get_data().astype('float32'))]
        else:
            if self.prev_im_ == im_:
                s = self.prev_input_data
                l = self.prev_labels
            else:
                s = [nib.load(
                        self.input_data[im_][k]).get_data().astype('float32')
                                for k in range(self.num_modalities)]
                l = [nib.load(
                        self.input_labels[im_][0]).get_data().astype('float32')]

        if self.normalize:
            s = [normalize_data(s[m], norm_type = self.norm_type) for m in range(len(s))]

        self.prev_input_data = s.copy()
        self.prev_labels = l.copy()

        # get current patches for both training data and labels
        input_train = np.stack([s[m][tuple(slice_)]
                                for m in range(self.num_modalities)], axis=0)
        input_label = np.expand_dims(
            l[0][tuple(slice_)], axis=0)

        # check dimensions and put zeros if necessary
        if input_train.shape != self.input_train_dim:
            print('error in patch', input_train.shape, self.input_train_dim)
            input_train = np.zeros(self.input_train_dim).astype('float32')
        if input_label.shape != self.input_label_dim:
            print('error in label', input_label.shape, self.input_label_dim)
            input_label = np.zeros(self.input_label_dim).astype('float32')

        if self.transform:
            input_train, input_label = self.transform([input_train,
                                                       input_label])

        self.prev_im_ = im_

        return input_train, input_label
            

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

        #TODO: Include information about case number (store dictionary instead of list in init)

        for i in range(len(self.input_data)): # Process one image at a time
            #Padding
            if self.pad_or_not:
                s = [self.apply_padding(nib.load(
                        self.input_data[i][k]).get_data().astype('float32'))
                              for k in range(self.num_modalities)]
                l = [self.apply_padding(nib.load(
                        self.input_labels[i][0]).get_data().astype('float32'))]
                r = [self.apply_padding(nib.load(
                        self.input_rois[i][0]).get_data().astype('float32'))]
            #No pading
            else:
                s = [nib.load(
                        self.input_data[i][k]).get_data().astype('float32')
                              for k in range(self.num_modalities)]
                l = [nib.load(
                        self.input_labels[i][0]).get_data().astype('float32')]
                r = [nib.load(
                        self.input_rois[i][0]).get_data().astype('float32')]                


            candidate_voxels = self.get_candidate_voxels(s[0], l[0], r[0]) #FLAIR, labels, brain mask
            voxel_coords = get_voxel_coordenates(s[0],
                                                 candidate_voxels,
                                                 step_size=self.sampling_step,
                                                 random_pad=self.random_pad)
            training_indexes += [(i, tuple(v)) for v in voxel_coords]

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


class PatchLoader3DLoadAll(Dataset):
    """
    Dataset class for loading MRI patches from multiple modalities. All patches are loaded before training. Based on script utils.py provided by Sergi Valverde. 

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 patch_size,
                 sampling_step,
                 random_pad=(0, 0, 0),
                 sampling_type='mask',
                 normalize=False,
                 norm_type='zero_one',
                 min_sampling_th=0,
                 num_pos_samples=5000,
                 resample_epoch=False,
                 transform=None):
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - patch_size: patch size
        - sampling_step: sampling_step
        - sampling type: 'all: all voxels in input_mask,
                            'roi: all voxels in roi_mask,
                            'balanced: same number of positive and negative voxels
        - normalize: Normalize data (0 mean / 1 std)
        - min_sampling_th: Minimum value to extract samples (0 default)
        - num_pos_samples used when hybrid sampling
        - transform
        """
        self.input_data = input_data
        self.input_labels = labels
        self.input_rois = rois
        self.pad_or_not = True
        self.patch_size = patch_size
        self.sampling_step = sampling_step
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
            self.all_patches = self.load_all_patches()

        return self.all_patches[idx, :, :, :, :], self.all_labels[idx,:,:,:,:]


            
    def load_all_patches(self):

        all_patches = np.zeros((len(self.patch_indexes), self.num_modalities, self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype='float32')
        all_labels = np.zeros((len(self.patch_indexes), len(list(self.input_labels.values())[0][0]), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype='float32')
        prev_pat = None
        prev_tp = None
        for idx in range(len(self.patch_indexes)):
            print(idx, "/", len(self.patch_indexes))

            im_ = self.patch_indexes[idx][0] # patient number
            tp = self.patch_indexes[idx][1] #Timepoint
            center = self.patch_indexes[idx][2] #Center of the patch

            slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                    for (c_idx, p_idx, s_idx) in zip(center,
                                                    self.patch_half,
                                                    self.patch_size)]

            #Read images only if different image index
            if (prev_pat != im_ or prev_tp != tp):
                if self.pad_or_not:
                    s = [self.apply_padding(nib.load(
                            self.input_data[im_][tp][k]).get_data().astype('float32'))
                                    for k in range(self.num_modalities)]
                    l = [self.apply_padding(nib.load(
                            self.input_labels[im_][tp][0]).get_data().astype('float32'))]
                else:
                    s = [nib.load(
                            self.input_data[im_][tp][k]).get_data().astype('float32')
                                    for k in range(self.num_modalities)]
                    l = [nib.load(
                            self.input_labels[im_][tp][0]).get_data().astype('float32')]

                if self.normalize:
                    s = [normalize_data(s[m], norm_type = self.norm_type) for m in range(len(s))]

                prev_pat = im_
                prev_tp = tp

            # get current patches for both training data and labels
            input_train = np.stack([s[m][tuple(slice_)]
                                    for m in range(self.num_modalities)], axis=0)
            input_label = np.expand_dims(
                l[0][tuple(slice_)], axis=0)

            # check dimensions and put zeros if necessary
            if input_train.shape != self.input_train_dim:
                print('error in patch', input_train.shape, self.input_train_dim)
                input_train = np.zeros(self.input_train_dim).astype('float32')
            if input_label.shape != self.input_label_dim:
                print('error in label', input_label.shape, self.input_label_dim)
                input_label = np.zeros(self.input_label_dim).astype('float32')

            if self.transform:
                input_train, input_label = self.transform([input_train,
                                                        input_label])

            #self.prev_im_ = im_

            all_patches[idx, :, :, :, :] = input_train.astype('float32')
            all_labels[idx, 0, :, :, :] = input_label.astype('float32')

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


                candidate_voxels = self.get_candidate_voxels(s[0], l[0], r[0]) #FLAIR, labels, brain mask
                voxel_coords = get_voxel_coordenates(s[0],
                                                    candidate_voxels,
                                                    step_size=self.sampling_step,
                                                    random_pad=self.random_pad)
                training_indexes += [(patient_number, tp, tuple(v)) for v in voxel_coords]

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


class PatchLoader3DTime(Dataset):
    """
    Dataset class for loading MRI patches from multiple modalities. Based on script utils.py provided by Sergi Valverde. 
    Difference with respect to PatchLoader3DTime: Timepoints are sliced so that more patches can be taken

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 patch_size,
                 sampling_step,
                 random_pad=(0, 0, 0),
                 sampling_type='mask',
                 normalize=False,
                 norm_type='zero_one',
                 min_sampling_th=0,
                 num_pos_samples=5000,
                 resample_epoch=False,
                 transform=None,
                 num_timepoints = 4):
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - patch_size: patch size
        - sampling_step: sampling_step
        - sampling type: 'all: all voxels in input_mask,
                            'roi: all voxels in roi_mask,
                            'balanced: same number of positive and negative voxels
        - normalize: Normalize data (0 mean / 1 std)
        - min_sampling_th: Minimum value to extract samples (0 default)
        - num_pos_samples used when hybrid sampling
        - transform
        """
        self.input_data = input_data
        self.input_labels = labels
        self.input_rois = rois
        self.pad_or_not = True
        self.patch_size = patch_size
        self.sampling_step = sampling_step
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
        self.num_timepoints = num_timepoints

        #Check that number of images coincide 
        if not len(input_data) == len(labels) == len(rois):
            raise ValueError("Number of input samples, input labels and input rois does not coincide")

        self.num_modalities = len(list(input_data.values())[0][0])
        self.input_train_dim = (self.num_timepoints, self.num_modalities, ) + self.patch_size
        self.input_label_dim = (self.num_timepoints, 1, ) + self.patch_size

        self.patch_indexes = self.generate_patch_indexes()


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

        im_ = self.patch_indexes[idx][0] #Patient
        slice_indexes = self.patch_indexes[idx][1] #Time slices
        center = self.patch_indexes[idx][2] #Center of the patch

        slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                  for (c_idx, p_idx, s_idx) in zip(center,
                                                   self.patch_half,
                                                   self.patch_size)]

        output_patch = np.zeros(self.input_train_dim, dtype = 'float32') #Array to store output patches
        output_label = np.zeros(self.input_label_dim, dtype = 'float32') #Array to store output labels
        ind = 0
        for i_t in slice_indexes: #For each timepoint
            #Read images -> Super slow, reads image for every patch
            if self.pad_or_not:
                s = [self.apply_padding(nib.load(
                        self.input_data[im_][i_t][k]).get_data().astype('float32'))
                                for k in range(self.num_modalities)]
                l = [self.apply_padding(nib.load(
                        self.input_labels[im_][i_t][0]).get_data().astype('float32'))]
            else:
                s = [nib.load(
                        self.input_data[im_][i_t][k]).get_data().astype('float32')
                                for k in range(self.num_modalities)]
                l = [nib.load(
                        self.input_labels[im_][i_t][0]).get_data().astype('float32')]

            if self.normalize:
                s = [normalize_data(s[m], norm_type = self.norm_type) for m in range(len(s))]


            # get current patches for both training data and labels
            input_train = np.stack([s[m][tuple(slice_)]
                                    for m in range(self.num_modalities)], axis=0)
            input_label = np.expand_dims(
                l[0][tuple(slice_)], axis=0)

            # check dimensions and put zeros if necessary
            if (self.num_timepoints,)+input_train.shape != self.input_train_dim:
                print('error in patch', input_train.shape, self.input_train_dim)
                input_train = np.zeros(self.input_train_dim).astype('float32')
            if (self.num_timepoints,)+input_label.shape != self.input_label_dim:
                print('error in label', input_label.shape, self.input_label_dim)
                input_label = np.zeros(self.input_label_dim).astype('float32')

            if self.transform:
                input_train, input_label = self.transform([input_train,
                                                        input_label])



            output_patch[ind,:,:,:] = input_train 
            output_label[ind,:,:,:] = input_label

            ind+=1

        return output_patch, output_label
            

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
            
            #Check that num_timepoints < len(timepoints_list)
            if not len(timepoints_list)> self.num_timepoints:
                continue # Ignore current patient, try with next one

            for i in range(len(timepoints_list) - self.num_timepoints + 1):
                #Padding
                if self.pad_or_not:
                    s = [self.apply_padding(nib.load(
                            timepoints_list[i][k]).get_data().astype('float32')) #Take first timepoint as reference for the patches
                                for k in range(self.num_modalities)]
                    l = [self.apply_padding(nib.load(
                            self.input_labels[patient_number][i][0]).get_data().astype('float32'))] #Take GT of last timepoint
                    r = [self.apply_padding(nib.load(
                            self.input_rois[patient_number][i][0]).get_data().astype('float32'))] #Take last brain mask 
                #No pading
                else:
                    s = [nib.load(
                            timepoints_list[i][k]).get_data().astype('float32')
                                for k in range(self.num_modalities)]
                    l = [nib.load(
                            self.input_labels[patient_number][i][0]).get_data().astype('float32')]
                    r = [nib.load(
                            self.input_rois[patient_number][i][0]).get_data().astype('float32')]                


                candidate_voxels = self.get_candidate_voxels(s[0], l[0], r[0]) #FLAIR, labels, brain mask
                voxel_coords = get_voxel_coordenates(s[0],
                                                    candidate_voxels,
                                                    step_size=self.sampling_step,
                                                    random_pad=self.random_pad)
                training_indexes += [(patient_number, tuple(range(i,i+self.num_timepoints)), tuple(v)) for v in voxel_coords]

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


class PatchLoader3DTimeLoadAll(Dataset):
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
                 sampling_step,
                 random_pad=(0, 0, 0),
                 sampling_type='mask',
                 normalize=False,
                 norm_type='zero_one',
                 min_sampling_th=0,
                 num_pos_samples=5000,
                 resample_epoch=False,
                 transform=None,
                 num_timepoints = 4,
                 labels_mode = 'mask'): # 'mask' 'center' or 'lesion_patch'
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - patch_size: patch size
        - sampling_step: sampling_step
        - sampling type: 'all: all voxels in input_mask,
                            'roi: all voxels in roi_mask,
                            'balanced: same number of positive and negative voxels
        - normalize: Normalize data (0 mean / 1 std)
        - min_sampling_th: Minimum value to extract samples (0 default)
        - num_pos_samples used when hybrid sampling
        - transform
        - num_timepoints: Number of timepoints to be considered
        - labels_mode: Type of label for the patches. If 'mask', the whole mask of the patch is returned
                        if 'center', only the value of the center pixel is returned as label. If 'lesion_patch'
                        then a label is returned which indicates whether or not the patch contains a lesion voxel.
        """
        self.input_data = input_data
        self.input_labels = labels
        self.input_rois = rois
        self.pad_or_not = True
        self.patch_size = patch_size
        self.sampling_step = sampling_step
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
        self.num_timepoints = num_timepoints
        self.labels_mode = labels_mode #To decide what the GT of the patches is: "mask", "lesion_patch" (if patch contains lesion), or "TODO"

        #Check that number of images coincide 
        if not len(input_data) == len(labels) == len(rois):
            raise ValueError("Number of input samples, input labels and input rois does not coincide")

        self.num_modalities = len(list(input_data.values())[0][0])
        self.input_train_dim = (self.num_timepoints, self.num_modalities, ) + self.patch_size
        self.input_label_dim = (self.num_timepoints, 1, ) + self.patch_size

        self.patch_indexes = self.generate_patch_indexes()
        self.all_patches, self.all_labels = self.load_all_patches()
        if self.labels_mode == 'lesion_patch':
            self.all_patches, self.all_labels = self.balance_data()

    def __len__(self):
        """
        Get the legnth of the training set
        """
        return len(self.all_patches)

    def __getitem__(self, idx):
        """
        Get the next item. Resampling the entire dataset is considered if
        self.resample_epoch is set to True.
        """

        if idx == 0 and self.resample_epoch:
            self.patch_indexes = self.generate_patch_indexes()
            self.all_patches, self.all_labels = self.load_all_patches()
        
        patches = self.all_patches[idx, :, :, :, :, :]
        if self.labels_mode == 'lesion_patch':
            labels = self.all_labels[idx]
        else:
            labels = self.all_labels[idx, :, np.newaxis,:,:,:]

        if self.transform:
            if self.labels_mode:
                patches = self.transform((patches))
                labels = torch.Tensor(labels)
            else:
                patches, labels = self.transform((patches, labels))

        return patches, labels
            

    def balance_data(self):
        num_samples = len(self.all_labels)
        num_pos = np.count_nonzero(self.all_labels)
        num_neg = num_samples - num_pos
        negative_indexes = np.where(self.all_labels == 0)[0]
        random.shuffle(negative_indexes) #Shuffle samples
        negative_indexes_to_remove = negative_indexes[:num_neg-num_pos] # Remove first num_neg-num_pos elements
        return np.delete(self.all_patches, negative_indexes_to_remove, 0), np.delete(self.all_labels, negative_indexes_to_remove, 0)
        

    def load_all_patches(self):

        all_patches = np.zeros((len(self.patch_indexes), self.num_timepoints, self.num_modalities, self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype='float32')
        if self.labels_mode == 'lesion_patch':
            output_labels = np.zeros((len(self.patch_indexes), ), dtype = np.uint8)

        all_labels = np.zeros((len(self.patch_indexes), self.num_timepoints, self.patch_size[0],self.patch_size[1],self.patch_size[2]), dtype=np.uint8)
        
        all_s = [] #To store images from all x timepoints required
        all_l = []
        prev_im_ = None
        prev_slice_indexes = None

        for idx in range(len(self.patch_indexes)):
            print(idx+1, "/", len(self.patch_indexes))
            im_ = self.patch_indexes[idx][0] #Patient
            slice_indexes = self.patch_indexes[idx][1] #Time slices
            center = self.patch_indexes[idx][2] #Center of the patch

            slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                    for (c_idx, p_idx, s_idx) in zip(center,
                                                    self.patch_half,
                                                    self.patch_size)]

            output_patch = np.zeros(self.input_train_dim, dtype = 'float32') #Array to store output patches
            output_label = np.zeros(self.input_label_dim, dtype = 'float32') #Array to store output labels
            ind = 0

            #Condition to load new images only if they are different as compared to those for previous idx
            if prev_im_ != im_ or prev_slice_indexes != slice_indexes: # If image or timepoints change
                all_s = []
                all_l = []

                for i_t in slice_indexes: #For each timepoint
                    #Read images -> Super slow, reads image for every patch

                    if self.pad_or_not:
                        all_s.append([self.apply_padding(nib.load(
                                self.input_data[im_][i_t][k]).get_data().astype('float32'))
                                        for k in range(self.num_modalities)])
                        all_l.append([self.apply_padding(nib.load(
                                self.input_labels[im_][i_t][0]).get_data().astype('float32'))])
                    else:
                        all_s.append([nib.load(
                                self.input_data[im_][i_t][k]).get_data().astype('float32')
                                        for k in range(self.num_modalities)])
                        all_l.append([nib.load(
                                self.input_labels[im_][i_t][0]).get_data().astype('float32')])

                if self.normalize:
                    for i_tp in range(len(all_s)):
                        all_s[i_tp] = [normalize_data(all_s[i_tp][m], norm_type = self.norm_type) for m in range(len(all_s[i_tp]))]

                prev_im_ = im_
                prev_slice_indexes = slice_indexes

            for i_t in range(len(all_s)):    

                # get current patches for both training data and labels
                input_train = np.stack([all_s[i_t][m][tuple(slice_)]
                                        for m in range(self.num_modalities)], axis=0)
                input_label = np.expand_dims(
                    all_l[i_t][0][tuple(slice_)], axis=0)

                # check dimensions and put zeros if necessary
                if (self.num_timepoints,)+input_train.shape != self.input_train_dim:
                    print('error in patch', input_train.shape, self.input_train_dim)
                    input_train = np.zeros(self.input_train_dim).astype('float32')
                if (self.num_timepoints,)+input_label.shape != self.input_label_dim:
                    print('error in label', input_label.shape, self.input_label_dim)
                    input_label = np.zeros(self.input_label_dim).astype('float32')

                if self.transform:
                    input_train, input_label = self.transform([input_train,
                                                            input_label])



                output_patch[ind,:,:,:,:] = input_train 
                output_label[ind,:,:,:,:] = input_label

                ind+=1

            all_patches[idx,:,:,:,:,:] = output_patch 
            all_labels[idx,:,:,:,:] = output_label[:,0,:,:,:]

            if self.labels_mode == 'lesion_patch':
                output_labels[idx] = int(np.any(all_labels[idx,-1,:,:,:]>0)) #If patch contains any positive voxel, return 1
        
        if self.labels_mode == 'lesion_patch':
            return all_patches, output_labels        
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
            
            #Check that num_timepoints < len(timepoints_list)
            if not len(timepoints_list)> self.num_timepoints:
                continue # Ignore current patient, try with next one

            for i in range(len(timepoints_list) - self.num_timepoints + 1):
                #Padding
                if self.pad_or_not:
                    s = [self.apply_padding(nib.load(
                            timepoints_list[i][k]).get_data().astype('float32')) #Take first timepoint as reference for the patches
                                for k in range(self.num_modalities)]
                    l = [self.apply_padding(nib.load(
                            self.input_labels[patient_number][i][0]).get_data().astype('float32'))] #Take GT of last timepoint
                    r = [self.apply_padding(nib.load(
                            self.input_rois[patient_number][i][0]).get_data().astype('float32'))] #Take last brain mask 
                #No pading
                else:
                    s = [nib.load(
                            timepoints_list[i][k]).get_data().astype('float32')
                                for k in range(self.num_modalities)]
                    l = [nib.load(
                            self.input_labels[patient_number][i][0]).get_data().astype('float32')]
                    r = [nib.load(
                            self.input_rois[patient_number][i][0]).get_data().astype('float32')]                


                candidate_voxels = self.get_candidate_voxels(s[0], l[0], r[0]) #FLAIR, labels, brain mask
                voxel_coords = get_voxel_coordenates(s[0],
                                                    candidate_voxels,
                                                    step_size=self.sampling_step,
                                                    random_pad=self.random_pad)
                training_indexes += [(patient_number, tuple(range(i,i+self.num_timepoints)), tuple(v)) for v in voxel_coords]

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




def extract_patches(input_image,
                    roi=None,
                    voxel_coords=None,
                    patch_size=(15, 15, 15),
                    step_size=(1, 1, 1)):
    """
    Extract patches of size patch_size from an input image given as input

    inputs:
    - input_image:  3D np.array
    - roi: region of interest to extract samples. input_image > 0 if not set
    - voxel_coords: Already computed voxel coordenades
    - patch_size: output patch size
    - step_size: sampling overlap in x, y and z

    output:
    - list of sampled patches: 4D array [n, patch_size] eg: (100, 15, 15 ,15)
    - list of voxel_coordenates
    """

    # check roi
    if roi is None:
        roi = input_image > 0

    # get voxel coordenates taking into account step sampling if those are
    # not passed as input
    if voxel_coords is None:
        voxel_coords = get_voxel_coordenates(input_image,
                                             roi,
                                             step_size=step_size)

    # extract patches based on the sampled voxel coordenates
    out_patches = get_patches(input_image, voxel_coords, patch_size)

    return out_patches, voxel_coords


def get_voxel_coordenates(input_data,
                          roi,
                          random_pad=(0, 0, 0),
                          step_size=(1, 1, 1)):
    """
    Get voxel coordenates based on a sampling step size or input mask.
    For each selected voxel, return its (x,y,z) coordinate.

    inputs:
    - input_data (useful for extracting non-zero voxels)
    - roi: region of interest to extract samples. input_data > 0 if not set
    - step_size: sampling overlap in x, y and z
    - random_pad: initial random padding applied to indexes

    output:
    - list of voxel coordenates
    """

    # compute initial padding
    r_pad = np.random.randint(random_pad[0]+1) if random_pad[0] > 0 else 0
    c_pad = np.random.randint(random_pad[1]+1) if random_pad[1] > 0 else 0
    s_pad = np.random.randint(random_pad[2]+1) if random_pad[2] > 0 else 0

    # precompute the sampling points based on the input
    sampled_data = np.zeros_like(input_data)
    for r in range(r_pad, input_data.shape[0], step_size[0]):
        for c in range(c_pad, input_data.shape[1], step_size[1]):
            for s in range(s_pad, input_data.shape[2], step_size[2]):
                sampled_data[r, c, s] = 1

    # apply sampled points to roi and extract sample coordenates
    # [x, y, z] = np.where(input_data * roi * sampled_data)
    [x, y, z] = np.where(roi * sampled_data)

    # return as a list of tuples
    return [(x_, y_, z_) for x_, y_, z_ in zip(x, y, z)]


def get_patches(input_data, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of voxel coordenates

    inputs:
    - input_data: a tridimensional np.array matrix
    - centers:  centre voxel coordenate for each patch
    - patch_size: patch size (x,y,z)

    outputs:
    - patches: np.array containing each of the patches
    """
    # If the size has even numbers, the patch will be centered. If not,
    # it will try to create an square almost centered. By doing this we allow
    # pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        # apply padding to the input image and re-compute the voxel coordenates
        # according to the new dimension
        padded_image = apply_padding(input_data, patch_size)
        patch_half = tuple([idx // 2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        # compute patch locations
        slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]

        # extact patches
        patches = [padded_image[tuple(idx)] for idx in slices]

    return np.array(patches)


def reconstruct_image(input_data, centers, output_size):
    """
    Reconstruct image based on several ovelapping patch samples

    inputs:
    - input_data: a np.array list with patches
    - centers: center voxel coordenates for each patch
    - output_size: output image size (x,y,z)

    outputs:
    - reconstructed image
    """

    # apply a padding around edges before writing the results
    # recompute the voxel dimensions
    patch_size = input_data[0, :].shape
    out_image = apply_padding(np.zeros(output_size), patch_size)
    patch_half = tuple([idx // 2 for idx in patch_size])
    new_centers = [map(add, center, patch_half) for center in centers]
    # compute patch locations
    slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
               for (c_idx, p_idx, s_idx) in zip(center,
                                                patch_half,
                                                patch_size)]
              for center in new_centers]

    # for each patch, sum it to the output patch and
    # then update the frequency matrix

    freq_count = np.zeros_like(out_image)

    for patch, slide in zip(input_data, slices):
        out_image[tuple(slide)] += patch
        freq_count[tuple(slide)] += np.ones(patch_size)

    # invert the padding applied for patch writing
    out_image = invert_padding(out_image, patch_size)
    freq_count = invert_padding(freq_count, patch_size)

    # the reconstructed image is the mean of all the patches
    out_image[freq_count!=0] = out_image[freq_count!=0]/freq_count[freq_count!=0]
    out_image[np.isnan(out_image)] = 0

    return out_image


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


def invert_padding(padded_image, patch_size):
    """
    Invert paadding on edges to recover the original shape

    inputs:
    - padded_image defined by apply_padding function
    - patch_size (x,y,z)

    """

    patch_half = tuple([idx // 2 for idx in patch_size])
    padding = tuple((idx, size-idx)
                    for idx, size in zip(patch_half, patch_size))

    return padded_image[padding[0][0]:-padding[0][1],
                        padding[1][0]:-padding[1][1],
                        padding[2][0]:-padding[2][1]]

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


def get_inference_patches(path_test, case, input_data, roi, patch_shape, step, normalize=True, norm_type = "zero_one", mode = "cs", num_timepoints = None):
    """
    Get patches for inference

    inputs:
    - scan path: path/to/the/subject to infer
    - input_data: list containing the input modality names
    - roi: ROI mask name
    - patch_shape: patch size
    - step: sampling step
    - normalize = zero mean normalization
    - norm_type: Type of normalization to be applied
    - mode: cross-sectional (cs) or longitudinal (l)

    outputs:
    - test patches (samples, channels, x, y, z)
    - ref voxels coordenates  extracted

    """
    
    if mode == "cs":
        scan_path = jp(path_test, case)
        # get candidate voxels
        timepoints = list_folders(scan_path)
        output_patches = []
        all_ref_voxels = []
        for tp in range(len(timepoints)):
            mask_image = nib.load(os.path.join(scan_path, timepoints[tp], roi))

            _, ref_voxels = get_candidate_voxels(mask_image.get_data(),
                                                        step,
                                                        sel_method='all')
            all_ref_voxels.append(ref_voxels)
            # input images stacked as channels
            patches = get_data_channels(os.path.join(scan_path, timepoints[tp]),
                                            input_data,
                                            ref_voxels,
                                            patch_shape,
                                            step,
                                            normalize=normalize,
                                            norm_type = norm_type)
            output_patches.append(patches)
        return output_patches, all_ref_voxels

    elif mode == "l":
        scan_path = jp(path_test, case)
        list_images = get_dictionary_with_paths([case], path_test, input_data) 
        list_rois = get_dictionary_with_paths([case], path_test, roi)

        brain_mask = list_rois[case][num_timepoints-1][0]   #ROI of last timepoint chosen
        mask_image = nib.load(os.path.join(scan_path, brain_mask))

        _, ref_voxels = get_candidate_voxels(mask_image.get_data(),
                                                    step,
                                                    sel_method='all')

        test_patches = get_data_channels_time(list_images,
                                        case,
                                        scan_path,
                                        input_data,
                                        ref_voxels,
                                        patch_shape,
                                        step,
                                        normalize=normalize,
                                        norm_type = norm_type)
        return test_patches, ref_voxels

    else:
        raise ValueError("Unknown mode.")

def get_data_channels_time( list_images,
                            case,
                            image_path,
                            scan_names,
                            ref_voxels,
                            patch_shape,
                            step,
                            normalize=False,
                            norm_type = "zero_one"):
    """
    Get data for each of the channels
    """
    super_out_patches = []
    for i in range(len(list_images[case])):
        out_patches = []
        for s in list_images[case][i]:
            current_scan = s
            patches, _ = get_input_patches(current_scan,
                                        ref_voxels,
                                        patch_shape,
                                        step,
                                        normalize=normalize,
                                        norm_type = norm_type)
            out_patches.append(np.expand_dims(patches, axis = 1 ))

        super_out_patches.append(np.concatenate(out_patches, axis=2))
    return np.concatenate(super_out_patches, axis=1)


def get_data_channels(image_path,
                      scan_names,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False,
                      norm_type = "zero_one"):
    """
    Get data for each of the channels
    """
    out_patches = []
    for s in scan_names:
        current_scan = os.path.join(image_path, s)
        patches, _ = get_input_patches(current_scan,
                                       ref_voxels,
                                       patch_shape,
                                       step,
                                       normalize=normalize,
                                       norm_type = norm_type)
        out_patches.append(patches)

    return np.concatenate(out_patches, axis=1)


def get_input_patches(scan_path,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False,
                      norm_type = 'zero_one',
                      expand_dims=True):
    """
    get current patches for a given scan
    """
    # current_scan = nib.as_closest_canonical(nib.load(scan_path)).get_data()
    current_scan = nib.load(scan_path).get_data()

    if normalize:
        current_scan = normalize_data(current_scan, norm_type = norm_type)

    patches, ref_voxels = extract_patches(current_scan,
                                          voxel_coords=ref_voxels,
                                          patch_size=patch_shape,
                                          step_size=step)

    if expand_dims:
        patches = np.expand_dims(patches, axis=1)

    return patches, ref_voxels


def get_candidate_voxels(input_mask,  step_size, sel_method='all'):
    """
    Extract candidate patches.
    """

    if sel_method == 'all':
        candidate_voxels = input_mask > 0

        voxel_coords = get_voxel_coordenates(input_mask,
                                             candidate_voxels,
                                             step_size=step_size)
    return candidate_voxels, voxel_coords


def build_image(infer_patches, lesion_model, device, num_classes, options):
  sh = infer_patches.shape
  lesion_out = np.zeros((sh[0], num_classes, sh[-3], sh[-2], sh[-1]))
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
  return lesion_out




