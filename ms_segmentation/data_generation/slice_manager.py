import os
import torch
import random
import nibabel as nib
import numpy as np
from .patch_manager_2d import normalize_data
from torch.utils.data import Dataset
from ..general.general import list_folders, cls, get_dictionary_with_paths
from os.path import join as jp





class SlicesGroupLoader(Dataset):
    """Slices group in terms of spatial context"""

    def __init__(self,
                 input_data,
                 labels,
                 roi,
                 num_slices, #Even number
                 out_size = (160,200),
                 normalize=True, # Whether or not the patches should be normalized
                 norm_type='zero_one',
                 transform=None): # Transforms to be applied to the patches

        self.input_data = list(input_data.values()) # Extract image paths from dictionary
        self.input_labels = list(labels.values()) # Extract labels paths from dictionary
        self.input_rois = list(roi.values())
        self.num_slices = num_slices
        self.normalize = normalize
        self.norm_type = norm_type
        self.transform = transform
        self.out_size = out_size

        self.num_modalities = len(list(input_data.values())[0])

        self.data = self.list_all()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.normalize: #Normalize whole image
            s = [normalize_data( nib.load(self.data[idx][0][k]).get_fdata().astype('float32'), norm_type = self.norm_type)
                        for k in range(self.num_modalities)]
        else:
            s = [nib.load(self.data[idx][0][k]).get_fdata().astype('float32')
                        for k in range(self.num_modalities)]


        l = [nib.load(
                self.data[idx][1][0]).get_fdata()[:,:,self.data[idx][2]].astype('float32')]

        images = np.zeros((self.num_slices, self.num_modalities, l[0].shape[0], l[0].shape[1]))
        pivot = self.num_slices // 2
        for i_mod in range(self.num_modalities):
            images[:,i_mod,:,:] = np.transpose(s[i_mod][:,:,self.data[idx][2]-pivot:self.data[idx][2]+pivot+1], (2,0,1))

        cropped_images, cropped_labels = self.crop_images(images, l[0][np.newaxis, :, :])

        return cropped_images, cropped_labels


    def crop_images(self, images, labels):
        """Function for cropping images to desired size
        
        Parameters
        ----------
        images : [type]
            [description]
        labels : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        center_x = labels.shape[-2] // 2 
        center_y = labels.shape[-1] // 2
        return images[:,:,center_x - self.out_size[0]//2: center_x + self.out_size[0]//2, center_y - self.out_size[1]//2: center_y + self.out_size[1]//2], \
                labels[:,center_x - self.out_size[0]//2: center_x + self.out_size[0]//2, center_y - self.out_size[1]//2: center_y + self.out_size[1]//2]
        


    def list_all(self):
        all_elements = []
        for i in range(len(self.input_data)): # Process one image at a time
            # read first image to get number of slices
            roi = nib.load(self.input_rois[i][0]).get_fdata()
            #total_slices = roi.shape[2]
            #total_slices = nib.load(self.input_data[i][0]).get_fdata().shape[2]
            lower_limit, upper_limit = self.get_limits(roi)
            #lower_limit = self.num_slices//2
            #upper_limit = total_slices - lower_limit - 1
            for j in range(lower_limit,upper_limit):    
                all_elements.append(([self.input_data[i][h] for h in range(self.num_modalities)], self.input_labels[i], j))

        return all_elements

    def list_folders(self,the_path):
        """Function to list folders in a specific path
        
        Parameters
        ----------
        the_path : string
            Path for which folders should be listed
        
        Returns
        -------
        List
            List of folder names
        """
        return [d for d in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, d))]

    def get_limits(self, roi):
        """Function to get indexes of first and last brain slices (where brain begins and where it ends)
        
        Parameters
        ----------
        roi : numpy array
            Brain mask
        
        Returns
        -------
        lower_limit: int
            Indexes of first brain slice according to the mask
        upper_limit: int
            Indexes of last brain slice according to the mask
        """
        for i in range(roi.shape[2]): #For each slice
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                lower_limit = i
                break
        for i in range(roi.shape[2]-1, 0, -1):
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                upper_limit = i
                break
        return lower_limit, upper_limit


#-------------------------------------------


class SlicesLoader(Dataset):
    """Slices."""

    def __init__(self,
                 input_data,
                 labels,
                 roi,
                 normalize=True, # Whether or not the patches should be normalized
                 norm_type='zero_one',
                 transform=None): # Transforms to be applied to the patches

        self.input_data = list(input_data.values()) # Extract image paths from dictionary
        self.input_labels = list(labels.values()) # Extract labels paths from dictionary
        self.normalize = normalize
        self.norm_type = norm_type
        self.transform = transform
        self.input_rois = list(roi.values())

        self.num_modalities = len(list(input_data.values())[0])

        self.data = self.list_all()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.normalize: #Wrong: Normalize whole image, not slice. AND DONT NORMALIZE LABELS
            s = [normalize_data( nib.load(self.data[idx][0][k]).get_fdata().astype('float32'), norm_type = self.norm_type)[:,:,self.data[idx][2]]
                        for k in range(self.num_modalities)]
            l = [normalize_data( nib.load(
                    self.data[idx][1][0]).get_fdata()[:,:,self.data[idx][2]].astype('float32'), norm_type = self.norm_type)]

        else:
            s = [nib.load(self.data[idx][0][k]).get_fdata().astype('float32')[:,:,self.data[idx][2]]
                        for k in range(self.num_modalities)]
            l = [nib.load(
                    self.data[idx][1][0]).get_fdata()[:,:,self.data[idx][2]].astype('float32')]


        images = np.zeros((self.num_modalities, l[0].shape[0], l[0].shape[1]))
        for i_mod in range(self.num_modalities):
            images[i_mod,:,:] = s[i_mod]

        return images, l[0][np.newaxis, :, :]

    def list_all(self):
        all_elements = []
        for i in range(len(self.input_data)): # Process one image at a time
            # read first image to get number of slices
            roi = nib.load(self.input_rois[i][0]).get_fdata()
            #total_slices = roi.shape[2]
            #total_slices = nib.load(self.input_data[i][0]).get_fdata().shape[2]
            lower_limit, upper_limit = self.get_limits(roi)
            #lower_limit = self.num_slices//2
            #upper_limit = total_slices - lower_limit - 1
            for j in range(lower_limit,upper_limit):    
                all_elements.append(([self.input_data[i][h] for h in range(self.num_modalities)], self.input_labels[i], j))

        return all_elements

    def list_folders(self,the_path):
        return [d for d in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, d))]

    def get_limits(self, roi):
        """Function to get indexes of first and last brain slices (where brain begins and where it ends)
        
        Parameters
        ----------
        roi : numpy array
            Brain mask
        
        Returns
        -------
        lower_limit: int
            Indexes of first brain slice according to the mask
        upper_limit: int
            Indexes of last brain slice according to the mask
        """
        for i in range(roi.shape[2]): #For each slice
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                lower_limit = i
                break
        for i in range(roi.shape[2]-1, 0, -1):
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                upper_limit = i
                break
        return lower_limit, upper_limit




class SlicesGroupLoaderTime(Dataset):
    """Slices group in terms of temporal context"""

    def __init__(self,
                 input_data,
                 labels,
                 roi,
                 num_timepoints, #Even number
                 out_size = (160,200),
                 normalize=True, # Whether or not the patches should be normalized
                 norm_type='zero_one',
                 transform=None): # Transforms to be applied to the patches

        self.input_data = input_data 
        self.input_labels = labels 
        self.input_rois = roi
        self.num_timepoints = num_timepoints
        self.normalize = normalize
        self.norm_type = norm_type
        self.transform = transform
        self.out_size = out_size

        self.num_modalities = len(list(input_data.values())[0][0])

        self.data = self.list_all()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        l = [nib.load(
                self.data[idx][2]).get_fdata()[:,:,self.data[idx][-1]].astype('float32')]

        images = np.zeros((self.num_timepoints, self.num_modalities, l[0].shape[0], l[0].shape[1]))

        if self.normalize:
            for i_t in range(self.num_timepoints):
                for i_m in range(self.num_modalities):
                    images[i_t, i_m, :, :] = normalize_data(nib.load(self.data[idx][1][i_t][i_m]).get_fdata().astype('float32'), norm_type = self.norm_type)[:,:,self.data[idx][-1]]
        else:
            for i_t in range(self.num_timepoints):
                for i_m in range(self.num_modalities):
                    images[i_t, i_m, :, :] = nib.load(self.data[idx][1][i_t][i_m]).get_fdata().astype('float32')[:,:,self.data[idx][-1]]


        cropped_images, cropped_labels = self.crop_images(images, l[0][np.newaxis, :, :])

        return cropped_images, cropped_labels


    def crop_images(self, images, labels):
        """Function for cropping images to desired size
        
        Parameters
        ----------
        images : [type]
            [description]
        labels : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        center_x = labels.shape[-2] // 2 
        center_y = labels.shape[-1] // 2
        return images[:,:,center_x - self.out_size[0]//2: center_x + self.out_size[0]//2, center_y - self.out_size[1]//2: center_y + self.out_size[1]//2], \
                labels[:,center_x - self.out_size[0]//2: center_x + self.out_size[0]//2, center_y - self.out_size[1]//2: center_y + self.out_size[1]//2]
        


    def list_all(self):
        all_elements = []
        for patient, timepoints_list in self.input_data.items(): # Process one image at a time
            
            for i in range(len(timepoints_list) - self.num_timepoints + 1): #For every possible combination of consecutive timepoints
                # read first image to get number of slices
                roi = nib.load(self.input_rois[patient][i + self.num_timepoints - 1][0]).get_fdata() #ROI of last timepoint of the group

                lower_limit, upper_limit = self.get_limits(roi)
                for j in range(lower_limit,upper_limit):    
                    #all_elements.append(([timepoints_list[h] for h in range(self.num_modalities)], self.input_labels[patient][i + self.num_timepoints -1],  j))
                    all_elements.append((patient, [timepoints_list[q] for q in range(i,i+self.num_timepoints)], self.input_labels[patient][i + self.num_timepoints -1][0],  j))

                #tuple(range(i,i+self.num_timepoints))

        return all_elements

    def list_folders(self,the_path):
        """Function to list folders in a specific path
        
        Parameters
        ----------
        the_path : string
            Path for which folders should be listed
        
        Returns
        -------
        List
            List of folder names
        """
        return [d for d in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, d))]

    def get_limits(self, roi):
        """Function to get indexes of first and last brain slices (where brain begins and where it ends)
        
        Parameters
        ----------
        roi : numpy array
            Brain mask
        
        Returns
        -------
        lower_limit: int
            Indexes of first brain slice according to the mask
        upper_limit: int
            Indexes of last brain slice according to the mask
        """
        for i in range(roi.shape[2]): #For each slice
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                lower_limit = i
                break
        for i in range(roi.shape[2]-1, 0, -1):
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                upper_limit = i
                break
        return lower_limit, upper_limit



class SlicesGroupLoaderTimeLoadAll(Dataset):
    """Slices group in terms of temporal context"""

    def __init__(self,
                 input_data,
                 labels,
                 roi,
                 num_timepoints, #Even number
                 out_size = (160,200),
                 normalize=True, # Whether or not the patches should be normalized
                 norm_type='zero_one',
                 transform=None): # Transforms to be applied to the patches

        self.input_data = input_data 
        self.input_labels = labels 
        self.input_rois = roi
        self.num_timepoints = num_timepoints
        self.normalize = normalize
        self.norm_type = norm_type
        self.transform = transform
        self.out_size = out_size

        self.num_modalities = len(list(input_data.values())[0][0])

        self.data = self.list_all()
        self.all_patches, self.all_labels = self.load_all_patches()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.all_patches[idx], self.all_labels[idx]

    def load_all_patches(self):
        
    
        all_patches = np.zeros((len(self.data), self.num_timepoints, self.num_modalities, self.out_size[0], self.out_size[1]), dtype = 'float32')
        all_labels = np.zeros((len(self.data), 1, self.out_size[0], self.out_size[1]), dtype = np.uint8)

        for idx in range(len(self.data)):
            cls()
            print("Loading all patches...")
            print(idx, "/", len(self.data))
            l = [nib.load(
                    self.data[idx][2]).get_fdata()[:,:,self.data[idx][-1]].astype('float32')]

            images = np.zeros((self.num_timepoints, self.num_modalities, l[0].shape[0], l[0].shape[1]))

            if self.normalize:
                for i_t in range(self.num_timepoints):
                    for i_m in range(self.num_modalities):
                        images[i_t, i_m, :, :] = normalize_data(nib.load(self.data[idx][1][i_t][i_m]).get_fdata().astype('float32'), norm_type = self.norm_type)[:,:,self.data[idx][-1]]
            else:
                for i_t in range(self.num_timepoints):
                    for i_m in range(self.num_modalities):
                        images[i_t, i_m, :, :] = nib.load(self.data[idx][1][i_t][i_m]).get_fdata().astype('float32')[:,:,self.data[idx][-1]]


            all_patches[idx], all_labels[idx] = self.crop_images(images, l[0][np.newaxis, :, :])

        return all_patches, all_labels


    def crop_images(self, images, labels):
        """Function for cropping images to desired size
        
        Parameters
        ----------
        images : [type]
            [description]
        labels : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        center_x = labels.shape[-2] // 2 
        center_y = labels.shape[-1] // 2
        return images[:,:,center_x - self.out_size[0]//2: center_x + self.out_size[0]//2, center_y - self.out_size[1]//2: center_y + self.out_size[1]//2], \
                labels[:,center_x - self.out_size[0]//2: center_x + self.out_size[0]//2, center_y - self.out_size[1]//2: center_y + self.out_size[1]//2]
        


    def list_all(self):
        all_elements = []
        for patient, timepoints_list in self.input_data.items(): # Process one image at a time
            
            for i in range(len(timepoints_list) - self.num_timepoints + 1): #For every possible combination of consecutive timepoints
                # read first image to get number of slices
                roi = nib.load(self.input_rois[patient][i + self.num_timepoints - 1][0]).get_fdata() #ROI of last timepoint of the group

                lower_limit, upper_limit = self.get_limits(roi)
                for j in range(lower_limit,upper_limit):    
                    #all_elements.append(([timepoints_list[h] for h in range(self.num_modalities)], self.input_labels[patient][i + self.num_timepoints -1],  j))
                    all_elements.append((patient, [timepoints_list[q] for q in range(i,i+self.num_timepoints)], self.input_labels[patient][i + self.num_timepoints -1][0],  j))

                #tuple(range(i,i+self.num_timepoints))

        return all_elements

    def list_folders(self,the_path):
        """Function to list folders in a specific path
        
        Parameters
        ----------
        the_path : string
            Path for which folders should be listed
        
        Returns
        -------
        List
            List of folder names
        """
        return [d for d in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, d))]

    def get_limits(self, roi):
        """Function to get indexes of first and last brain slices (where brain begins and where it ends)
        
        Parameters
        ----------
        roi : numpy array
            Brain mask
        
        Returns
        -------
        lower_limit: int
            Indexes of first brain slice according to the mask
        upper_limit: int
            Indexes of last brain slice according to the mask
        """
        for i in range(roi.shape[2]): #For each slice
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                lower_limit = i
                break
        for i in range(roi.shape[2]-1, 0, -1):
            curr_slice = roi[:,:,i]
            if(np.count_nonzero(curr_slice)>0):
                upper_limit = i
                break
        return lower_limit, upper_limit





def get_inference_slices(scan_path, input_data, normalize=True, norm_type = 'zero_one'):

    if normalize: 
        s = [normalize_data( nib.load(jp(scan_path, mod)).get_fdata().astype('float32'), norm_type = norm_type)
                    for mod in input_data]
    else:
        s = [nib.load(jp(scan_path, mod)).get_fdata().astype('float32')
                    for mod in input_data]

    num_modalities = len(input_data)
    num_slices = s[0].shape[-1]
    h = s[0].shape[0]
    w = s[0].shape[1]

    images = np.zeros((num_slices, num_modalities, h, w))
    for i_mod in range(num_modalities):
        images[:,i_mod,:,:] = np.transpose(s[i_mod], (2,0,1))

    return images

def get_inference_slices_time(the_path, the_case, input_data, out_size, normalize = True, norm_type='zero_one'):
    """Function to return the inference patches with dimension (num_slices, num_timepoints, modalities, height, width)
    
    Parameters
    ----------
    the_path : str
        [description]
    the_case : str
        [description]
    input_data : list
        [description]
    out_size : tuple
        [description]
    normalize : bool, optional
        [description], by default True
    norm_type : str, optional
        [description], by default 'zero_one'
    
    Returns
    -------
    [type]
        [description]
    """
    list_images = get_dictionary_with_paths([the_case], the_path, input_data) 
    num_timepoints = len(list_images[the_case])

    #Get number of slices from first image of first time point
    dim_x, dim_y, num_slices = nib.load(list_images[the_case][0][0]).get_fdata().shape

    all_slices = np.zeros((num_slices, num_timepoints, len(input_data), dim_x, dim_y), dtype = "float32")
    for t in range(len(list_images[the_case])): #Iterate in tmepoints
        for mod in range(len(list_images[the_case][t])): #Iterate in modalities
            if normalize:
                all_slices[:,t, mod,:,:] = np.transpose(normalize_data(nib.load(list_images[the_case][t][mod]).get_fdata(), norm_type=norm_type), (2,0,1))
            else:
                all_slices[:,t, mod,:,:] = np.transpose(nib.load(list_images[the_case][t][mod]).get_fdata(), (2,0,1))


    all_slices = crop_images(all_slices, out_size)

    return all_slices


def crop_images(images, out_size):
    """Function for cropping images to desired size
    
    Parameters
    ----------
    images : [type]
        [description]
    labels : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    center_x = images.shape[-2] // 2 
    center_y = images.shape[-1] // 2
    return images[:,:,:,center_x - out_size[0]//2: center_x + out_size[0]//2, center_y - out_size[1]//2: center_y + out_size[1]//2]

def undo_crop_images(big_image, cropped_labels, out_size_cropped):
    """Function to undo cropping done previously. Important: Cropping was made on last two dimensions (slice dimensions). For
    this function the slice dimensions are at the beginning
    
    Parameters
    ----------
    big_image : [type]
        [description]
    cropped_labels : [type]
        [description]
    out_size_cropped : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    center_x = big_image.shape[0] // 2 
    center_y = big_image.shape[1] // 2
    big_image[center_x - out_size_cropped[0]//2: center_x + out_size_cropped[0]//2, center_y - out_size_cropped[1]//2: center_y + out_size_cropped[1]//2, :] = cropped_labels

    return big_image

def get_probs(inf_slices, lesion_model, device, num_classes, options):
    #lesion_out = np.zeros_like(infer_patches).astype('float32')
    sh = inf_slices.shape
    lesion_out = np.zeros((sh[0], num_classes, sh[-2], sh[-1]))
    batch_size = options['batch_size']
    b =0
    # model
    lesion_model.eval()
    with torch.no_grad():
        for b in range(0, len(lesion_out), batch_size):
            x = torch.tensor(inf_slices[b:b+batch_size]).type('torch.cuda.FloatTensor').to(device)
            pred = lesion_model(x)
            # save the result back from GPU to CPU --> numpy
            lesion_out[b:b+batch_size] = pred.detach().cpu().numpy()
    return lesion_out