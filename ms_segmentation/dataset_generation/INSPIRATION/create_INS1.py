# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script to randomly divide the cross-sectional dataset (CROSS_SECTIONAL2) into training and testing sets with 70% and 30%, respectively
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------

import os
import random
import numpy as np
from os.path import join as jp
from distutils.dir_util import copy_tree
from general.general import list_folders, create_folder

path_data = r'D:\dev\CROSS_SECTIONAL2'
destiny = r'D:\dev\INS1' # Where I want to copy the images (new dataset)

create_folder(destiny)
create_folder(jp(destiny, "train"))
create_folder(jp(destiny, "test"))

percentage_train = 0.7 # Train + validation
percentage_test =  0.3 # Test

all_cases_list = list_folders(path_data)
# First, shuffle list of cases
random.shuffle(all_cases_list)

pivot = int(np.ceil(len(all_cases_list)*percentage_train))

for i in range(len(all_cases_list)):
    print(i, "/", len(all_cases_list))
    if i <= pivot:
        #Copy to train
        create_folder(jp(destiny, "train", all_cases_list[i])) #Create folder in destiny
        copy_tree(jp(path_data, all_cases_list[i]), jp(destiny, "train", all_cases_list[i]))
    else:
        #Copy to test
        create_folder(jp(destiny, "test", all_cases_list[i]))
        copy_tree(jp(path_data, all_cases_list[i]), jp(destiny, "test", all_cases_list[i]))
        



