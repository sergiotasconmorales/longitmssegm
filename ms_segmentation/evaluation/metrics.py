# --------------------------------------------------------------------------------------------------------------------
#
# Project:      MS lesion segmentation (master thesis)
#
# Description:  Script with segmentation metrics such as DSC, FTP, VD, etc
#
# Author:       Sergio Tascon Morales (Research intern at mediri GmbH, student of Master in Medical Imaging and Applications - MAIA)
#
# Details:      None
#
# --------------------------------------------------------------------------------------------------------------------


import numpy as np 
import SimpleITK as sitk


def compute_dices(gt, pred):
  ''' Function to calculate DSC for two vectors 
  '''
  if(np.max(gt)==0):
    return np.array([0.0], dtype = float)
  dices = np.zeros((np.max(gt),), dtype=float)
  for i in range(np.max(gt)):
    gt_bool = np.zeros_like(gt, dtype = np.bool)
    segm_bool = np.zeros_like(pred, dtype = np.bool)
    gt_bool[gt==i+1] = True
    segm_bool[pred==i+1] = True
    if(not np.sum(gt_bool)+ np.sum(segm_bool)):
      dices[i] = 1.0
      continue
    dices[i] = (2. * np.sum(gt_bool * segm_bool)) / (np.sum(gt_bool) + np.sum(segm_bool))
  return dices


#Hausdorf distance
def compute_hausdorf(gt, pred):
  if(np.max(gt)==0 or np.count_nonzero(pred)==0):
    return np.array([10000], dtype=float)
  h_distances = np.zeros((np.max(gt),), dtype=float)
  for i in range(np.max(gt)):
    hd = sitk.HausdorffDistanceImageFilter()
    gt_new = np.zeros_like(gt, dtype = np.uint8)
    segm_new = np.zeros_like(pred, dtype = np.uint8)
    gt_new[gt==i+1] = 1
    segm_new[pred==i+1] = 1
    hd.Execute(sitk.GetImageFromArray(gt_new), sitk.GetImageFromArray(segm_new))
    h_distances[i] = hd.GetHausdorffDistance()
  return h_distances


