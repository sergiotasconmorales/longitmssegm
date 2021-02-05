import os
import pandas as pd
import nibabel as nib 
import numpy as np
from os.path import join as jp
from ms_segmentation.general.general import list_folders, list_files_with_extension
from ms_segmentation.evaluation.metrics import compute_metrics 




reference = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs\qwertz06'
to_compare = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images\results_cs\qwertz09'
comparison_id = reference.split("\\")[-1] + "__" + to_compare.split("\\")[-1] 
save_path = r'D:\dev\ms_data\Challenges\ISBI2015\Test_Images'

columns = compute_metrics(None, None, labels_only=True)
df = pd.DataFrame(columns = columns)

list_ref = list_files_with_extension(reference, "nii")
list_to_compare = list_files_with_extension(to_compare, "nii")

assert len(list_ref) == len(list_to_compare)
i_row = 0
for i,img in enumerate(list_ref):
    print(i+1, "/", len(list_ref))
    ref_labels = nib.load(jp(reference, img)).get_fdata().astype(np.uint8)
    to_compare_labels = nib.load(jp(to_compare, img)).get_fdata().astype(np.uint8)

    metrics = compute_metrics(ref_labels, to_compare_labels)
    df.loc[i_row] = list(metrics.values())
    i_row += 1

df.loc[i_row] = list(df.mean())
df.to_csv(jp(save_path, comparison_id + ".csv"), float_format = '%.5f', index = False)