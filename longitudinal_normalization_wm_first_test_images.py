import cv2
from os.path import join as jp
import nibabel as nib
import numpy as np
from scipy import ndimage
import os
from copy import deepcopy as dcopy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import local_binary_pattern as lbp
from ms_segmentation.general.general import save_image, list_folders, create_folder, list_files_with_name_containing
from ms_segmentation.data_generation.patch_manager_3d import normalize_data
from scipy.optimize import fmin

path_data = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\longitudinal'
path_histograms = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\chi_square_histograms'
path_write = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\chi_square_images'
create_folder(path_write)
create_folder(path_histograms)

patients = list_folders(path_data)

modalities = ["flair", "mprage", "pd", "t2"]
#tissues = ["wm", "gm", "mask1"]
tissues = ["brain_mask"]
to_analyze = {'preprocessed': 'nii'}
colors = ['b', 'g', 'r', 'c', 'm'] 
ref = []
wm_masks_ref = []
#f = lambda x: np.sum((np.histogram(fl_0[fl_0>0].ravel(), 256, [0,1], density=True)[0] - np.histogram(x[0]*fl_1[fl_1>0].ravel(), 256, [0,1], density=True)[0]**2)/(np.histogram(fl_0[fl_0>0].ravel(), 256, [0,1], density=True)[0] + 0.00001))
image_format = "nii.gz"


all_histograms = [[], [], [], []]
all_histograms_aligned = [[], [], [], []]
all_histograms_wm = [[], [], [], []]
all_histograms_gm = [[], [], [], []]
all_histograms_mask = [[], [], [], []]
all_histograms_wm_aligned = [[], [], [], []]
all_histograms_gm_aligned = [[], [], [], []]
all_histograms_mask_aligned = [[], [], [], []]
for pat in patients: # for each patient
    create_folder(jp(path_write, pat))
    create_folder(jp(path_histograms, pat))
    print("Patient:", pat)
    all_paths = [list_files_with_name_containing(jp(path_data, pat), mod, image_format) for mod in modalities] # list (modality) of lists(tp)
    all_labels_paths = [list_files_with_name_containing(jp(path_data, pat), tiss, image_format) for tiss in tissues] # list (modality) of lists(tp)
    num_tp = len(all_paths[0])
    all_images = [[normalize_data(nib.load(p).get_fdata()) for p in all_paths[i]] for i in range(len(all_paths))] # list (modality) of lists(tp)
    all_labels = [[nib.load(l).get_fdata() for l in all_labels_paths[i]] for i in range(len(all_labels_paths))]
    #all_images_filtered_wm = [[normalize_data(anisodiff3(img*lbl, niter = 5)) for img, lbl in zip(all_images[i], all_labels[0])] for i in range(len(all_images))]
    
    histograms = [[np.histogram(img[img>0].ravel(), 256, [0,1], density=True) for img in all_images[i]] for i in range(len(all_images))]
    #histograms_anisotropic = [[np.histogram(img[img>0].ravel(), 256, [0,1], density=True) for img in all_images_filtered_wm[i]] for i in range(len(all_images_filtered_wm))]
    histograms_wm = [[np.histogram(img[lbl>0].ravel(), 256, [0,1], density=True) for img, lbl in zip(all_images[i], all_labels[0])] for i in range(len(all_images))]
    #histograms_gm = [[np.histogram(img[lbl>0].ravel(), 256, [0,1], density=True) for img, lbl in zip(all_images[i], all_labels[1])] for i in range(len(all_images))]
    #histograms_mask = [[np.histogram(img[lbl>0].ravel(), 256, [0,1], density=True) for img, lbl in zip(all_images[i], all_labels[2])] for i in range(len(all_images))]


    histograms_aligned = dcopy(histograms)
    all_images_aligned = dcopy(all_images)

    

    for i_mod in range(len(all_images)): # for each modality  
        if pat == "01":
            ref.append(all_images[i_mod][0])
            wm_masks_ref.append(all_labels[0][0]) # although it's the same for all timepoints
        curr_ref = ref[i_mod] #Always first timepoint of first patient as reference
        curr_ref_mask = wm_masks_ref[i_mod]
        #save_image(curr_ref, jp(path_write, pat, modalities[i_mod] + "_norm_" + str(0+1).zfill(2)))
        #histograms_aligned[i_mod][0] = histograms[i_mod][0] # First timepoint as reference(does not change)
        for i_tp in range(0, num_tp): # for each timepoint starting at the second one
            curr_img = all_images[i_mod][i_tp]
            curr_wm_mask = all_labels[0][i_tp]
            f = lambda x: cv2.compareHist(np.histogram(curr_ref[curr_ref_mask>0].ravel(), 256, [0,1], density=True)[0].astype(np.float32), 
                                        np.histogram((x[0]*curr_img[curr_wm_mask>0]).ravel(), 256, [0,1], density=True)[0].astype(np.float32), 1)
            # Optimize Chi-Square metric
            xopt = fmin(func = f, x0 = [1])
            curr_img_new = np.clip(xopt[0]*curr_img,0,1)
            save_image(curr_img_new, jp(path_write, pat, modalities[i_mod] + "_norm_" + str(i_tp+1).zfill(2) + ".nii.gz"))
            all_images_aligned[i_mod][i_tp] = curr_img_new
            histograms_aligned[i_mod][i_tp] = np.histogram(curr_img_new[curr_img_new>0].ravel(), 256, [0,1], density=True)
            
    # generate histograms after alignment        
    histograms_wm_aligned = [[np.histogram(img[lbl>0].ravel(), 256, [0,1], density=True) for img, lbl in zip(all_images_aligned[i], all_labels[0])] for i in range(len(all_images_aligned))]
    #histograms_gm_aligned = [[np.histogram(img[lbl>0].ravel(), 256, [0,1], density=True) for img, lbl in zip(all_images_aligned[i], all_labels[1])] for i in range(len(all_images_aligned))]
    #histograms_mask_aligned = [[np.histogram(img[lbl>0].ravel(), 256, [0,1], density=True) for img, lbl in zip(all_images_aligned[i], all_labels[2])] for i in range(len(all_images_aligned))]

    #all_images_aligned = [[np.clip(displace_hist(img, p, new_centers[i]), 0, 1) for img, p in zip(all_images[i], peaks[i])] for i in range(len(all_images))]

    #histograms_aligned = [[np.histogram(img[img>0].ravel(), 256, [0,1], density=True) for img in all_images_aligned[i]] for i in range(len(all_images_aligned))]

    
    for i_mod, mod in enumerate(histograms): # each modality
        all_histograms[i_mod].append(mod)
        create_folder(jp(path_histograms, pat, modalities[i_mod]))
        plt.figure()
        for hist in mod: # each timepoint
            plt.plot(hist[1][:-1], hist[0])
        plt.ylim([0,10])
        plt.grid()
        plt.savefig(jp(path_histograms, pat, modalities[i_mod], "hist.png"))

    """
    for i_mod, mod in enumerate(zip(histograms_mask, histograms_wm, histograms_gm)): # each modality
                create_folder(jp(path_histograms, pat, modalities[i_mod]))
                all_histograms_wm[i_mod].append(mod[1])
                all_histograms_gm[i_mod].append(mod[2])
                all_histograms_mask[i_mod].append(mod[0])
                plt.figure()
                for h_mask, h_wm, h_gm in zip(mod[0], mod[1], mod[2]): # each timepoint
                    plt.plot(h_mask[1][:-1], h_mask[0], 'k')
                    plt.plot(h_mask[1][:-1], h_wm[0], 'g')
                    plt.plot(h_mask[1][:-1], h_gm[0], 'b')
                plt.ylim([0,20])
                plt.legend(['mask', 'wm', 'gm'])
                plt.grid()
                plt.savefig(jp(path_histograms, pat, modalities[i_mod], "hist_tissues.png"))
    """
    for i_mod, mod in enumerate(histograms_aligned): # each modality
        all_histograms_aligned[i_mod].append(mod)
        create_folder(jp(path_histograms, pat, modalities[i_mod]))
        plt.figure()
        for hist in mod: # each timepoint
            plt.plot(hist[1][:-1], hist[0])
        plt.ylim([0,10])
        plt.grid()
        plt.savefig(jp(path_histograms, pat, modalities[i_mod], "hist_chi.png"))

    """
    for i_mod, mod in enumerate(zip(histograms_mask_aligned, histograms_wm_aligned, histograms_gm_aligned)): # each modality
                create_folder(jp(path_histograms, pat, modalities[i_mod]))
                all_histograms_wm_aligned[i_mod].append(mod[1])
                all_histograms_gm_aligned[i_mod].append(mod[2])
                all_histograms_mask_aligned[i_mod].append(mod[0])
                plt.figure()
                for h_mask, h_wm, h_gm in zip(mod[0], mod[1], mod[2]): # each timepoint
                    plt.plot(h_mask[1][:-1], h_mask[0], 'k')
                    plt.plot(h_mask[1][:-1], h_wm[0], 'g')
                    plt.plot(h_mask[1][:-1], h_gm[0], 'b')
                plt.ylim([0,20])
                plt.legend(['mask', 'wm', 'gm'])
                plt.grid()
                plt.savefig(jp(path_histograms, pat, modalities[i_mod], "hist_tissues_aligned.png"))
    """

#Plot all histograms
for i_mod, mod in enumerate(modalities):
    plt.figure()
    for i_pat, hists_pat in enumerate(all_histograms[i_mod]):
        curr_color = colors[i_pat]
        for curr_hist in hists_pat:
            plt.plot(curr_hist[1][:-1], curr_hist[0], curr_color)
    plt.grid()
    plt.savefig(jp(path_histograms, "hist" + mod + ".png"))

    plt.figure()
    for i_pat, hists_pat in enumerate(all_histograms_aligned[i_mod]):
        curr_color = colors[i_pat]
        for curr_hist in hists_pat:
            plt.plot(curr_hist[1][:-1], curr_hist[0], curr_color)
    plt.grid()
    plt.savefig(jp(path_histograms, "hist_chi_" + mod + ".png"))  

    #Plot tissue histograms
    """
    plt.figure()
    for i_pat, hists_pat in enumerate(zip(all_histograms_mask[i_mod], all_histograms_wm[i_mod], all_histograms_gm[i_mod])):
        for h_mask, h_wm, h_gm in zip(hists_pat[0], hists_pat[1], hists_pat[2]): # each timepoint
            plt.plot(h_mask[1][:-1], h_mask[0], 'k')
            plt.plot(h_mask[1][:-1], h_wm[0], 'g')
            plt.plot(h_mask[1][:-1], h_gm[0], 'b')
        plt.ylim([0,20])
        plt.legend(['mask', 'wm', 'gm'])
        plt.grid()
        plt.savefig(jp(path_histograms, pat, modalities[i_mod], "hist_chi_tissues" + mod + ".png")) 

    plt.figure()
    for i_pat, hists_pat in enumerate(zip(all_histograms_mask_aligned[i_mod], all_histograms_wm_aligned[i_mod], all_histograms_gm_aligned[i_mod])):
        for h_mask, h_wm, h_gm in zip(hists_pat[0], hists_pat[1], hists_pat[2]): # each timepoint
            plt.plot(h_mask[1][:-1], h_mask[0], 'k')
            plt.plot(h_mask[1][:-1], h_wm[0], 'g')
            plt.plot(h_mask[1][:-1], h_gm[0], 'b')
        plt.ylim([0,20])
        plt.legend(['mask', 'wm', 'gm'])
        plt.grid()
        plt.savefig(jp(path_histograms, pat, modalities[i_mod], "hist_chi_tissues_aligned" + mod + ".png")) 
    """