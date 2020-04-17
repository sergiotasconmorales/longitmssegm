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
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import confusion_matrix
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc

#Dice score
def compute_dices(gt, pred):
  """Function to compute the Dice score (DSC) between two 3d binary volumes
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  [type]
      [description]
  """
  # transform volumes to vectors
  gt = gt.flatten()
  pred = pred.flatten()
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
  """Function to compute hausdorff distance between two 3d binary volumes
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  list
      List with HD for each of the input classes
  """
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

def compute_jaccard(gt, pred):
  """Function to compute Jaccard coefficient between two binary volumes
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  float
      Value of the Jaccard index
  """
  jaccard = jsc(pred.flatten(), gt.flatten(), average = 'binary')
  return jaccard

def compute_tpr(gt, pred):
  """Function to compute voxel-wise TPR between two binary volumes
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  float
      TPR
  """
  _, _, fn, tp = confusion_matrix(gt.flatten(), pred.flatten()).ravel()
  return tp/(tp+fn)

def compute_fpr(gt, pred):
  """Function to compute voxel-wise FPR between two binary volumes
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  float
      FPR
  """
  tn, fp, _, _ = confusion_matrix(gt.flatten(), pred.flatten()).ravel()
  return fp/(fp+tn)

def compute_ppv(gt, pred):
  """Function to compute the positive predictive value from two binary segmentations
  
  Parameters
  ----------
  gt : [type]
      [description]
  pred : [type]
      [description]
  
  Returns
  -------
  [type]
      [description]
  """
  _, fp, _, tp = confusion_matrix(gt.flatten(), pred.flatten()).ravel()
  return tp/(tp+fp)

def compute_volumetric_difference(gt, pred):
  """Function to compute the volumetric difference between two segmentation results. Only for binary masks
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  float
      Value of the VD
  """
  vol_gt = np.count_nonzero(gt)
  vol_pred = np.count_nonzero(pred)
  return 100*(np.abs(vol_pred - vol_gt)/vol_gt)


def compute_f2_score(gt, pred):
  """Function to compute F2 score between two binary masks. Definition according to hashemi2018
  
  Parameters
  ----------
  gt : numpy array
      Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
      Predicted segmentation. Must have dtype=np.uint8

  Returns
  -------
  float
      Value of the F2 score
  """
  tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten()).ravel()
  return 5*tp/(5*tp + 4*fn + fp)


def num_regions(mask):
  """Function to get the number of regions or connected components
  By Sergi Valverde
  
  Parameters
  ----------
  mask : numpy array
      [description]
  
  Returns
  -------
  [type]
      [description]
  """
  _, num_regions = label(mask.astype(np.bool))
  return num_regions


def compute_ltp(gt, pred):
  """compute the number of positive regions between a input mask an a ground truth (GT) mask
  By Sergi Valverde
    
  Parameters
  ----------
  gt : numpy array
    Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
    Predicted segmentation. Must have dtype=np.uint8
    
  Returns
  -------
  int
  lesion-wise true positives
  """
  regions, num_regions = label(gt.astype(np.bool))
  labels = np.arange(1, num_regions+1)
  pred = pred.astype(np.bool)
  tpr = lc(pred, regions, labels, np.sum, int, 0)

  return np.sum(tpr > 0)

def compute_lfp(gt, pred):
  """compute the number of false positive lesions between a input mask an a ground truth (GT) mask.
  By Sergi Valverde
  
  Parameters
  ----------
  gt : numpy array
    Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
    Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  int
      lesion-wise false positives
  """
  regions, num_regions = label(pred.astype(np.bool))
  labels = np.arange(1, num_regions+1)
  gt = gt.astype(np.bool)

  return np.sum(lc(gt, regions, labels, np.sum, int, 0) == 0) \
      if num_regions > 0 else 0


def compute_ltpr(gt, pred):
  """Function to compute lesion-wise true positive rate (LTPR)
  By Sergi Valverde
  
  Parameters
  ----------
  gt : numpy array
    Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
    Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  [type]
      [description]
  """

  TP = compute_ltp(gt, pred)
  number_of_regions = num_regions(gt)

  return float(TP) / number_of_regions


def compute_lfpr(gt, pred):
  """Function to compute lesion-wise false positive rate (LFPR)
  By Sergi Valverde
    
  Parameters
  ----------
  gt : numpy array
    Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
    Predicted segmentation. Must have dtype=np.uint8
    
  Returns
  -------
  [type]
      [description]
  """

  FP = compute_lfp(gt, pred)
  number_of_regions = num_regions(pred)

  return float(FP) / number_of_regions if number_of_regions > 0 else 0






def compute_metrics(gt, pred, labels_only = False):
  """Function to compute all metrics and return them in a dictionary
  
  Parameters
  ----------
  gt : numpy array
    Ground truth volume. Must have dtype=np.uint8
  pred : numpy array
    Predicted segmentation. Must have dtype=np.uint8
  
  Returns
  -------
  dict
      dictionary with all metrics
  """
  if labels_only:
    return ["DSC","JACCARD","HD","TPR","FPR", "PPV", "AVD","F2","LTPR","LFPR"]
  metrics = {}
  metrics["DSC"] = compute_dices(gt, pred)[0]
  metrics["JACCARD"] = compute_jaccard(gt, pred)
  metrics["HD"] = compute_hausdorf(gt, pred)[0]
  metrics["TPR"] = compute_tpr(gt, pred)
  metrics["FPR"] = compute_fpr(gt, pred)
  metrics["PPV"] = compute_ppv(gt, pred)
  metrics["AVD"] = compute_volumetric_difference(gt, pred)
  metrics["F2"] = compute_f2_score(gt, pred)
  metrics["LTPR"] = compute_ltpr(gt, pred)
  metrics["LFPR"] = compute_lfpr(gt, pred)

  return metrics