# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script containing basic functions for list processing and information display
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------

import os
import glob
import nibabel as nib
import numpy as np
from datetime import datetime
from os.path import join as jp

# GENERAL FUNCTIONS -------------------------------------------------------------------------------------

def remove_from_list_if_contains(the_list, char):
    """Function to remove elements from list <the_list> that contain sequence of characters <char>
    
    Parameters
    ----------
    the_list : list
        input list to be modified
    char : string
        list with removed elements
    
    Returns
    -------
    list
        List with elements that contain <char> removed
    """

    return [elem for elem in the_list if char not in elem]


def list_folders(the_path):
    """Function to list folders only for a specific path
    
    Parameters
    ----------
    the_path : string
        Path for which folders should be listed
    
    Returns
    -------
    list
        List of folders of path <the_path>
    """
    return [d for d in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, d))]


def filter_list_of_names(file_names):
    """Function to filter the names of a list, removing elements that contain a dot
    
    Parameters
    ----------
    file_names : list
        A list of strings
    
    Returns
    -------
    list
        List with elements that do not contain a dot
    """
    return [h for h in file_names if '.' not in h]

def filter_list(the_list, the_strings):
    """Function to filter list. Only elements containing one of the elements of <the_strings> are in output
    
    Parameters
    ----------
    the_list : [type]
        [description]
    the_strings : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    if isinstance(the_strings, list):
        return [k for u in the_strings for k in the_list if u in k]
    else:
        return [k for k in the_list if the_strings in k]

def list_files_with_extension(the_path, extension):
    """Function to list all files with a certain extension in path <the_path>
    
    Parameters
    ----------
    the_path : string
        Path containing files
    extension : string
        Extension of the files that should be listed (only the name of the extension)
    
    Returns
    -------
    it : list
        List of files that have the extension <extension>
    """
    saved = os.getcwd()
    os.chdir(the_path)
    it = glob.glob('*.' + extension)
    os.chdir(saved)
    return it


def create_folder(the_path):
    """Function to create a folder if it does not exist
    
    Parameters
    ----------
    the_path : string
        Path of the folder to be created
    """
    if not os.path.exists(the_path):
        os.mkdir(the_path)
    else:
        pass

def get_experiment_name(the_prefix = "exp"):
    """Function to create an experiment name based on a prefix and time information (date + current time). Format is <the_prefix>_<date>_<time>
    
    Parameters
    ----------
    the_prefix : str, optional
        Value of the prefix for the experiment name, by default "exp"
    
    Returns
    -------
    experiment_name : str
        Name of the experiment
    the_date : str
        Current date
    the_time : str
        Current time
    """
    now = datetime.now()
    the_date = str(datetime.date(now))
    the_time = now.strftime("%H_%M_%S")
    experiment_name = the_prefix + "_" + the_date + "_" + the_time
    return experiment_name, the_date, the_time

def find_element_in_list_that_contains(the_list, the_string):
    """Function to return first element of <the_list> that contains <the_string>
    
    Parameters
    ----------
    the_list : list
        Input list
    the_string : string
        String to be found in elements of <the_list>
    
    Returns
    -------
    string or empty list
        Output of the function. If element was found, element is returned, if not found, empty list is returned
    """

    elems = [elem for elem in the_list if the_string in elem]
    if bool(elems):
        return elems[0]
    else:
        return elems

def list_files_with_name_containing(the_path, the_string, the_format):
    """Function to list all files that contain <the_string>
    
    Parameters
    ----------
    the_path : [type]
        [description]
    the_string : [type]
        [description]
    the_format : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    files = list_files_with_extension(the_path, the_format)
    return [os.path.join(the_path, f) for f in files if the_string in f]

def get_dictionary_with_paths(scans, the_path, the_names):
    """[summary]
    
    Parameters
    ----------
    scans : [type]
        scan (folder) names. In this case it corresponds to the patient index because it`s longitudinal
    the_path : str
        path to the images
    the_names : list or str
        names of the images that should be contained in the names of the files (eg ['flair', 'mprage'])
    
    Returns
    -------
    d : dictionary
        Dictionary with paths to the images 
    """
    d = {}
    for scan in scans:
            d[scan] = []
            num_time_points = len(list_files_with_name_containing(os.path.join(the_path, scan), "brain_mask", "nii.gz")) #Flair is always there, so use it as reference
            for i_t in range(num_time_points):
                d[scan].append(filter_list(list_files_with_name_containing(os.path.join(the_path, scan), str(i_t+1).zfill(2), "nii.gz"), the_names))
    return d

def get_dictionary_with_paths_cs(patients, the_path, the_names):
    
    d = {}
    for patient in patients:
        d[patient] = []
        timepoints = list_folders(jp(the_path, patient))
        d[patient] = [[jp(the_path, patient, tp, image) for image in the_names] for tp in timepoints] 

    return d

def print_line():
    """ Function to print a line in the console
    """
    print("-------------------------------------------------------------------")

def expand_dictionary(input_dict):
    output_dict = {}
    for k0,v0 in input_dict.items():
        curr_dict = v0
        temp_dict = {}
        for k1,v1 in curr_dict.items():
            cnt=1
            for tp in range(len(v1)):
                temp_dict[k1 + "_" + str(cnt).zfill(2)] = v1[tp]
                cnt+=1
        output_dict[k0] = temp_dict
    return output_dict

def cls():
    """Function to clear the console
    """
    os.system('cls||clear')

def save_image(the_array, the_path, orientation="RAI"):
    """Function to save a numpy array as an image. Name and format are specified in <the_path>
    
    Parameters
    ----------
    the_array : numpy array
        Image to be saved
    the_path : str
        Path where image should be saved (including image name and format)
    """
    if orientation == "LPI":
        img_nib = nib.Nifti1Image(the_array, np.eye(4))
    elif orientation == "RAI":
        img_nib = nib.Nifti1Image(the_array, np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]))
    nib.save(img_nib, the_path)

def create_log(path_results, options, filename = "log.txt"):
    f = open(jp(path_results, filename), "w+")
    for k,v in options.items():
        f.write("%s : %s\n" % (k,v))
    f.close()

# -------------------------------------------------------------------------------------------------------