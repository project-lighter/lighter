from pathlib import Path

from monai.transforms import RandSpatialCrop
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from loguru import logger

from .utils import normalization, sitk_utils
from .utils.misc import get_bounding_box_of_mask, pad
from .common import get_dataset_split


class NLSTDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, label, hounsfield_units_range,
                 mode, patch_size=None, transform=None):

        assert mode in ["train", "tune", "test"]
        self.mode = mode
        self.root_dir = Path(root_dir)
        split = get_dataset_split(self.root_dir / "SelectionTrainTestFinal.csv")
        self.mask_paths = [self.root_dir / "lung_masks" / image for image in split[mode]]
        self.mask_paths = sorted(list(filter(lambda path: path.is_file(), self.mask_paths)))

        self.nlst_labels = pd.read_csv(self.root_dir / "NLST_clinical_whole.csv")
        self.romans_labels = pd.read_csv(self.root_dir / "SelectionTrainTestFinal.csv")

        self.label = label
        self.hu_min, self.hu_max = hounsfield_units_range
        self.transform = transform
        self.patch_size = patch_size
        self.random_patch_sampler = RandSpatialCrop(self.patch_size,
                                                    random_size=False)
        self.mode = mode

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        scan_path = Path(str(mask_path).replace("lung_masks", "nrrd"))
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
        target = self.get_target(patient_id, self.label)

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
        if self.mode == "train":
            tensor = self.random_patch_sampler(tensor)
            # Pad to match the patch size if the resulting patch is smaller
            tensor = pad(tensor, self.patch_size)
        return tensor, target

    def get_target(self, patient_id, label):
        patient_id = int(patient_id)

        if label == "packyear":
            packyear = self.nlst_labels[self.nlst_labels.pid == int(patient_id)][label]
            packyear = float(packyear)

            NEVER, LIGHT, MODERATE, HEAVY = 0, 1, 2, 3
            if packyear > 40:
                return HEAVY
            if packyear > 20:
                return MODERATE
            if packyear > 1:
                return LIGHT
            return NEVER

        elif label == "age":
            age = self.nlst_labels[self.nlst_labels.pid == patient_id][label]
            return float(age)

        # Mortality prediction
        elif label == "fup_days":
            N_YEARS = 6
            fup_days = self.nlst_labels[self.nlst_labels.pid == patient_id][label]
            death = self.romans_labels[self.romans_labels.Patient_ID == patient_id]["Death"]
            if int(fup_days) < 365.25 * N_YEARS and int(death) == 1:
                return 1
            return 0

        raise NotImplementedError

    def get_labels(self):
        labels = []
        for path in self.mask_paths:
            patient_id = path.stem.replace("_img", "")
            labels.append(self.get_target(patient_id, self.label))
        return labels

    def __len__(self):
        return len(self.mask_paths)
