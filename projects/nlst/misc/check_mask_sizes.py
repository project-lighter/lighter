import pandas as pd
import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

sys.path.append("/home/ibrahim/projects/lighter/projects/nlst/")
from datasets.utils import misc, sitk_utils

mask_sizes = {}
masks_dir = Path("/mnt/ssd1/NLST/lung_masks/")
for path in tqdm(list(masks_dir.glob("[!.]*/*.nrrd"))):
    try:
        mask = sitk.ReadImage(str(path))
    except RuntimeError:
        print("Failed to read", str(path))
        continue
    try:
        bb = misc.get_bounding_box_of_mask(mask)
    except RuntimeError:
        print("No mask in", str(path))
        continue
    size = tuple(np.subtract(bb[1], bb[0]))
    mask_sizes[str(path)] = size

with open('mask_sizes.pkl', 'wb') as fp:
    pickle.dump(mask_sizes, fp)
