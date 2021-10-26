from pathlib import Path

import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from loguru import logger

from .utils import normalization, sitk_utils
from .utils.misc import get_bounding_box_of_mask, pad


class NLSTDataset:

    def __init__(self,
                 scans_dir,
                 masks_dir,
                 labels_path,
                 label,
                 patch_size,
                 hounsfield_units_range,
                 transform=None):

        self.mask_paths = list(Path(masks_dir).glob(f"[!.]*/*.nrrd"))
        self.scans_dir = Path(scans_dir)
        self.labels_df = pd.read_csv(labels_path)
        self.label = label

        self.random_patch_sampler = monai.transforms.RandSpatialCrop(patch_size, random_size=False)
        self.patch_size = patch_size
        self.hu_min, self.hu_max = hounsfield_units_range
        self.transform = transform

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        scan_path = self.scans_dir / mask_path.parent.name / mask_path.name

        # Load the scan
        try:
            scan = sitk_utils.load(scan_path)
        except RuntimeError:
            logger.info(f"Failed to load scan {scan_path}, skipping it.")
            return None

        # Load the mask
        try:
            mask = sitk_utils.load(mask_path)
        except RuntimeError:
            logger.info(f"Failed to load mask {mask_path}, skipping it.")
            return None

        patient_id = scan_path.stem.replace("_img", "")
        target = self.get_target(patient_id)

        # Originally, mask has a label for each lung, here we combine them into a single label
        mask = sitk.Clamp(mask, upperBound=1)
        # Mask out everything except lungs
        try:
            scan = sitk_utils.apply_mask(scan, mask, masking_value=0, outside_value=self.hu_min)
        except RuntimeError:
            logger.info(f"Scan ({scan_path}) and mask ({mask_path}) don't match, skipping.")
            return None

        # Get the bounding box of the mask
        try:
            start, end = get_bounding_box_of_mask(mask)
        except RuntimeError:
            logger.info(f"No mask in {mask_path}, skipping.")
            return None

        # Crop the masked scan using the bounding box
        scan = sitk_utils.slice_image(scan, start=start, end=end)

        # Lungs need to be at least 50 slices and at least 150x150
        min_size = np.array([50, 150, 150])
        current_size = sitk_utils.get_torch_like_size(scan)
        if (current_size < min_size).any():
            logger.info(f"Segmented lungs are smaller than the threshold size of {min_size}.")
            return None

        tensor = sitk_utils.get_tensor(scan)
        tensor = torch.clamp(tensor, self.hu_min, self.hu_max)
        tensor = normalization.min_max_normalize(tensor, self.hu_min, self.hu_max)
        # Add channel dimension (CDHW)
        tensor = tensor.unsqueeze(0)
        tensor = tensor if self.transform is None else self.transform(tensor)
        tensor = self.random_patch_sampler(tensor)
        # Pad to match the patch size if the resulting patch is smaller
        tensor = pad(tensor, self.patch_size)
        return tensor, target

    def get_target(self, patient_id):
        label_value = self.labels_df[self.labels_df.pid == int(patient_id)][self.label]

        if self.label == "packyear":
            NEVER, LIGHT, MODERATE, HEAVY = 0, 1, 2, 3
            packyear = float(label_value)
            if packyear > 40:
                return HEAVY
            if packyear > 20:
                return MODERATE
            if packyear > 1:
                return LIGHT
            return NEVER

        raise NotImplementedError

    def __len__(self):
        return len(self.mask_paths)
