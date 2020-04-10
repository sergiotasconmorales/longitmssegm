import os
import nibabel as nib
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from operator import add 
from ..general.general import list_folders
from os.path import join as jp


#----------------------------------------------------------------------------------------------------------------------
# 2D patches
#----------------------------------------------------------------------------------------------------------------------


class PatchLoader2D_slow(Dataset):
    """
    Dataset class for loading MRI patches from multiple modalities. Based on script utils.py provided by Sergi Valverde. 
    Each patch is loaded when it is needed, therefore it is slow. Use class PatchLoader2D (below) for a faster performance.

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois, # Brain mask
                 patch_size, # Desired size for the 2D patches
                 sampling_step, # Stride of the patches
                 random_pad=(0, 0), # Random pad for selecting the location of the patches
                 sampling_type='mask', # Type of sampling of the patches
                 normalize=False, # Whether or not the patches should be normalized
                 norm_type='zero_one',
                 min_sampling_th=0, # 
                 num_pos_samples=5000, # Maximum number of samples
                 resample_epoch=False, # Whether or not to resample after each epoch
                 transform=None): # Transforms to be applied to the patches

        self.input_data = list(input_data.values()) # Extract image paths from dictionary
        self.input_labels = list(labels.values()) # Extract labels paths from dictionary
        self.input_rois = list(rois.values()) # Extract brain mask paths from dictionary
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
                  for (c_idx, p_idx, s_idx) in zip(center[:-1],
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

        if self.normalize: #Normalize image
            s = [normalize_data(s[m], norm_type = self.norm_type) for m in range(len(s))]

        self.prev_input_data = s.copy()
        self.prev_labels = l.copy()

        # get current patches for both training data and labels
        input_train = np.stack([s[m][:,:,center[2]][tuple(slice_)]
                                for m in range(self.num_modalities)], axis=0)
        input_label = np.expand_dims(
            l[0][:,:,center[2]][tuple(slice_)], axis=0)

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

    # def remove_percentage(self, percentage):
    #     list_int = random.sample(range(len(self.patch_indexes)), int(percentage*len(self.patch_indexes)))
    #     return [self.patch_indexes[i] for i in list_int]

    def generate_patch_indexes(self):
        """
        Generate indexes to extract. Consider the sampling step and
        a initial random padding
        """
        training_indexes = []

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
                                                 random_pad=self.random_pad,
                                                 uniform = True)   #ACHTUUUUUUNG
            training_indexes += [(i, tuple(v)) for v in voxel_coords]

        print("Total number of patches: ", len(training_indexes))

        return training_indexes


    def get_candidate_voxels(self, input_image, label_mask, roi_mask):
        """
        Sample input mask using different techniques:
        - all: extracts all voxels > 0 from the input_image
        - mask: extracts all roi voxels
        - label: extracts all labels > 0
        - balanced: same number of positive and negative voxels from
                    the input_image as defined by the roi mask
        - balanced+roi: same number of positive and negative voxels from
                    the roi and label mask

        """

        if self.sampling_type == 'all': # ALl voxels greater than 0 in original image (brain voxels)
            sampled_mask = input_image > 0

        if self.sampling_type == 'label' or self.sampling_type == 'non-uniform': # All positive voxels of the labels
            sampled_mask = label_mask > 0

        if self.sampling_type == 'mask': # All voxels greater than zero in ROI (brain mask)
            sampled_mask = roi_mask > 0

        if self.sampling_type == 'balanced': 
            sampled_mask = label_mask > 0 # Voxels of the labels (lesion pixels) - BOOL
            num_positive = np.sum(label_mask > 0) # Number of positive voxels in label mask
            brain_voxels = np.stack(np.where(input_image > self.min_th), axis=1) # Coordenates of non-zero (brain) voxels in original image. ROI could be used too
            for voxel in np.random.permutation(brain_voxels)[:num_positive]: #Select first num_positive random brain voxels (some of them could overlap with positives from labels)
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        if self.sampling_type == 'balanced+roi':
            sampled_mask = label_mask > 0 #Voxels of the labels (lesion pixels)
            num_positive = np.sum(label_mask > 0) # Number of positive voxels in label mask
            roi_mask[label_mask == 1] = 0 # Make locations of GT zero in ROI (ROI - labels)
            brain_voxels = np.stack(np.where(roi_mask > 0), axis=1) # Coordenates of non-zero voxels in ROI
            for voxel in np.random.permutation(brain_voxels)[:num_positive]: # Select first num_positive random ROI voxels (no overlap with labels)
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        return sampled_mask


    def apply_padding(self, input_data, mode='constant', value=0):
        """
        Apply padding to edges in order to avoid overflow

        """

        #Apply padding only to first two dimensions
        padding = ((self.patch_half[0], self.patch_size[0] - self.patch_half[0]),(self.patch_half[1], self.patch_size[1] - self.patch_half[1]),(0,0))
        #padding = tuple((half, size-half)
        #                for half, size in zip(self.patch_half, self.patch_size))

        padded_image = np.pad(input_data,
                              padding,
                              mode=mode,
                              constant_values=value)
        return padded_image










class PatchLoader2D(Dataset):
    """
    Dataset class for loading MRI patches from multiple modalities. Based on script utils.py provided by Sergi Valverde. 
    All patches are loaded at once before __getitem__ is called. This is the change with respect to patch_manager_2d.py
    """

    def __init__(self,
                 input_data,
                 labels,
                 rois, # Brain mask
                 patch_size, # Desired size for the 2D patches
                 sampling_step, # Stride of the patches
                 random_pad=(0, 0), # Random pad for selecting the location of the patches
                 sampling_type='mask', # Type of sampling of the patches
                 normalize=False, # Whether or not the patches should be normalized
                 norm_type='zero_one',
                 min_sampling_th=0, # 
                 num_pos_samples=5000, # Maximum number of samples
                 resample_epoch=False, # Whether or not to resample after each epoch
                 transform=None): # Transforms to be applied to the patches

        self.input_data = list(input_data.values()) # Extract image paths from dictionary
        self.input_labels = list(labels.values()) # Extract labels paths from dictionary
        self.input_rois = list(rois.values()) # Extract brain mask paths from dictionary
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
        self.all_patches = self.load_all_patches()

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

        return self.all_patches[idx,:-1,:,:], self.all_patches[idx,-1,:,:][np.newaxis,:,:]


    def load_all_patches(self):

        # all_patches = [num_patches, num_modalities + 1, patch_side, patch_side]
        all_patches = np.zeros((len(self.patch_indexes), len(self.input_data[0]) + len(self.input_labels[0]), self.patch_size[0], self.patch_size[1]), dtype='float32')

        for idx in range(len(self.patch_indexes)):
            print(idx, "/", len(self.patch_indexes))
            im_ = self.patch_indexes[idx][0]
            center = self.patch_indexes[idx][1]

            slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                    for (c_idx, p_idx, s_idx) in zip(center[:-1],
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

            if self.normalize: #Normalize image
                s = [normalize_data(s[m], norm_type = self.norm_type) for m in range(len(s))]

            self.prev_input_data = s.copy()
            self.prev_labels = l.copy()

            # get current patches for both training data and labels
            input_train = np.stack([s[m][:,:,center[2]][tuple(slice_)]
                                    for m in range(self.num_modalities)], axis=0)
            input_label = np.expand_dims(
                l[0][:,:,center[2]][tuple(slice_)], axis=0)

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

            all_patches[idx,:-1, :, :] = input_train.astype('float32')
            all_patches[idx,-1, :, :] = input_label.astype('float32')

        return all_patches
    # def remove_percentage(self, percentage):
    #     list_int = random.sample(range(len(self.patch_indexes)), int(percentage*len(self.patch_indexes)))
    #     return [self.patch_indexes[i] for i in list_int]

    def generate_patch_indexes(self):
        """
        Generate indexes to extract. Consider the sampling step and
        a initial random padding
        """
        training_indexes = []

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
                                                 random_pad=self.random_pad,
                                                 uniform = True)   #ACHTUUUUUUNG
            training_indexes += [(i, tuple(v)) for v in voxel_coords]

        print("Total number of patches: ", len(training_indexes))

        return training_indexes


    def get_candidate_voxels(self, input_image, label_mask, roi_mask):
        """
        Sample input mask using different techniques:
        - all: extracts all voxels > 0 from the input_image
        - mask: extracts all roi voxels
        - label: extracts all labels > 0
        - balanced: same number of positive and negative voxels from
                    the input_image as defined by the roi mask
        - balanced+roi: same number of positive and negative voxels from
                    the roi and label mask

        """

        if self.sampling_type == 'all': # ALl voxels greater than 0 in original image (brain voxels)
            sampled_mask = input_image > 0

        if self.sampling_type == 'label' or self.sampling_type == 'non-uniform': # All positive voxels of the labels
            sampled_mask = label_mask > 0

        if self.sampling_type == 'mask': # All voxels greater than zero in ROI (brain mask)
            sampled_mask = roi_mask > 0

        if self.sampling_type == 'balanced': 
            sampled_mask = label_mask > 0 # Voxels of the labels (lesion pixels) - BOOL
            num_positive = np.sum(label_mask > 0) # Number of positive voxels in label mask
            brain_voxels = np.stack(np.where(input_image > self.min_th), axis=1) # Coordenates of non-zero (brain) voxels in original image. ROI could be used too
            for voxel in np.random.permutation(brain_voxels)[:num_positive]: #Select first num_positive random brain voxels (some of them could overlap with positives from labels)
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        if self.sampling_type == 'balanced+roi':
            sampled_mask = label_mask > 0 #Voxels of the labels (lesion pixels)
            num_positive = np.sum(label_mask > 0) # Number of positive voxels in label mask
            roi_mask[label_mask == 1] = 0 # Make locations of GT zero in ROI (ROI - labels)
            brain_voxels = np.stack(np.where(roi_mask > 0), axis=1) # Coordenates of non-zero voxels in ROI
            for voxel in np.random.permutation(brain_voxels)[:num_positive]: # Select first num_positive random ROI voxels (no overlap with labels)
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        return sampled_mask


    def apply_padding(self, input_data, mode='constant', value=0):
        """
        Apply padding to edges in order to avoid overflow

        """

        #Apply padding only to first two dimensions
        padding = ((self.patch_half[0], self.patch_size[0] - self.patch_half[0]),(self.patch_half[1], self.patch_size[1] - self.patch_half[1]),(0,0))
        #padding = tuple((half, size-half)
        #                for half, size in zip(self.patch_half, self.patch_size))

        padded_image = np.pad(input_data,
                              padding,
                              mode=mode,
                              constant_values=value)
        return padded_image


# Auxiliar functions

def get_voxel_coordenates(input_data,
                          roi,
                          random_pad=(0, 0),
                          step_size=(1, 1), 
                          uniform = True):
    """
    Get voxel coordenates based on a sampling step size or input mask.
    For each selected voxel, return its (x,y,z) coordinate.

    inputs:
    - input_data (useful for extracting non-zero voxels)
    - roi: region of interest to extract samples. input_data > 0 if not set
    - step_size: sampling overlap in x, y and z
    - random_pad: initial random padding applied to indexes
    - uniform: old way of selecting patches or new one (non-uniform)

    output:
    - list of voxel coordenates
    """
    if uniform:
        dims = [0,1,2] 
        #TODO: Provide posibility of extracting patches in sagittal and coronal orientations
        #TODO: Add another way of generating x,y,z so that more voxels of <roi> are considered

        # compute initial padding
        r_pad = np.random.randint(random_pad[0]+1) if random_pad[0] > 0 else 0
        c_pad = np.random.randint(random_pad[1]+1) if random_pad[1] > 0 else 0
        #s_pad = np.random.randint(random_pad[2]+1) if random_pad[2] > 0 else 0

        # precompute the sampling points for each axial slice
        sampled_data = np.zeros_like(input_data) # Mask of centroids of patches
        for i_slice in range(input_data.shape[2]):
            for r in range(r_pad, input_data.shape[0], step_size[0]):
                for c in range(c_pad, input_data.shape[1], step_size[1]):
                        sampled_data[r, c, i_slice] = 1

        # apply sampled points to roi and extract sample coordenates
        # [x, y, z] = np.where(input_data * roi * sampled_data)
        [x, y, z] = np.where(roi * sampled_data)

        # prod = roi*sampled_data
        # nib_img = nib.Nifti1Image(prod.astype(np.uint8), np.eye(4))
        # nib.save(nib_img, "prod_" + str(i) + ".nii.gz")
        # nib_brain = nib.Nifti1Image(brain.astype(np.uint8), np.eye(4))
        # nib.save(nib_brain, "brain_" + str(i) + ".nii.gz")

        # return as a list of tuples
        return [(x_, y_, z_) for x_, y_, z_ in zip(x, y, z)]
    else:
        pseudo_roi = input_data>0 #Brain mask approximation
        sampled_data = np.zeros_like(input_data)
        positives = np.stack(np.where(roi), axis=1)
        chosen_pos = np.random.permutation(positives)[:int(0.1*positives.shape[0]),:]
        pseudo_roi[chosen_pos[:,0], chosen_pos[:,1], chosen_pos[:,2]] = 0
        negatives = np.stack(np.where(pseudo_roi), axis=1)
        chosen_neg = np.random.permutation(negatives)[:chosen_pos.shape[0],:]
        all_ = np.random.permutation(np.concatenate((chosen_pos, chosen_neg), axis=0))

        # return as a list of tuples
        return [(x_, y_, z_) for x_, y_, z_ in all_]


def normalize_data(im,
                   norm_type='standard',
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
            im = (im.astype(dtype=datatype) - min_int) / (max_int - min_int) # Changed from max_int to (max_int - min_int) in denominator

    # do not apply normalization to non-brain parts
    # im[mask==0] = 0
    return im


def get_inference_patches(scan_path, input_data, roi, patch_shape, step, normalize=True, norm_type = 'zero_one'):
    """
    Get patches for inference

    inputs:
    - scan path: path/to/the/subject to infer
    - input_data: list containing the input modality names
    - roi: ROI mask name
    - patch_shape: patch size
    - step: sampling step
    - normalize = zero mean normalization

    outputs:
    - test patches (samples, channels, x, y, z)
    - ref voxels coordenates  extracted

    """


    # get candidate voxels
    mask_image = nib.load(os.path.join(scan_path, roi))

    ref_mask, ref_voxels = get_candidate_voxels(mask_image.get_data(),
                                                step,
                                                sel_method='all')

    # input images stacked as channels
    test_patches = get_data_channels(scan_path, # Path to test image
                                     input_data, # Modality names
                                     ref_voxels, # Locations of the candidates
                                     patch_shape, #Patch size
                                     step, # Patch step
                                     normalize=normalize,
                                     norm_type=norm_type)


    return test_patches, ref_voxels


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


def get_data_channels(image_path,
                      scan_names,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False,
                      norm_type = 'zero_one'):
    """
    Get data for each of the channels
    """
    out_patches = []
    for s in scan_names: # For each modality
        current_scan = os.path.join(image_path, s)
        patches, _ = get_input_patches(current_scan,  #Location of the patches is being discarded
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
        new_centers = [map(add, center, patch_half) for center in centers[:-1]]
        # compute patch locations

        slice_locations = [elem[2] for elem in centers]

        slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]

        # extact patches
        patches = [padded_image[:,:,slice_idx][tuple(idx)] for idx,slice_idx in zip(slices, slice_locations)]

    return np.array(patches)


def build_image(infer_patches, lesion_model, device, num_classes, options):
    #lesion_out = np.zeros_like(infer_patches).astype('float32')
    sh = infer_patches.shape
    lesion_out = np.zeros((sh[0], num_classes, sh[2], sh[3]))
    batch_size = options['batch_size']
    b =0
    # model
    lesion_model.eval()
    with torch.no_grad():
        for b in range(0, len(lesion_out), batch_size):
            x = torch.tensor(infer_patches[b:b+batch_size]).to(device)
            pred = lesion_model(x)
            # save the result back from GPU to CPU --> numpy
            lesion_out[b:b+batch_size] = pred.detach().cpu().numpy()
    return lesion_out



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
    new_centers = [map(add, center, patch_half) for center in centers[:-1]]
    # compute patch locations

    slice_locations = [elem[2] for elem in centers]

    slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                for (c_idx, p_idx, s_idx) in zip(center,
                                                patch_half,
                                                patch_size)]
                  for center in new_centers]

    # for each patch, sum it to the output patch and
    # then update the frequency matrix

    freq_count = np.zeros_like(out_image)


    for patch, slide, sl in zip(input_data, slices, slice_locations):
        out_image[:,:,sl][tuple(slide)] += patch
        freq_count[:,:,sl][tuple(slide)] += np.ones(patch_size)

    # invert the padding applied for patch writing
    out_image = invert_padding(out_image, patch_size)
    freq_count = invert_padding(freq_count, patch_size)

    # the reconstructed image is the mean of all the patches
    #out_image /= freq_count
    out_image[freq_count!=0] = out_image[freq_count!=0]/freq_count[freq_count!=0]
    out_image[np.isnan(out_image)] = 0

    return out_image


def apply_padding(input_data, patch_size, mode='constant', value=0):
    """
    Apply padding to edges in order to avoid overflow

    """

    patch_half = tuple([idx // 2 for idx in patch_size])
    #padding = tuple((idx, size-idx)
    #                for idx, size in zip(patch_half, patch_size))
    padding = ((patch_half[0], patch_size[0] - patch_half[0]), (patch_half[1], patch_size[1] - patch_half[1]),(0,0))

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
    - patch_size (x,y)

    """

    patch_half = tuple([idx // 2 for idx in patch_size])
    padding = tuple((idx, size-idx)
                    for idx, size in zip(patch_half, patch_size))

    return padded_image[padding[0][0]:-padding[0][1],
                        padding[1][0]:-padding[1][1]]




