from pathlib import Path

import monai
import pandas as pd
import SimpleITK as sitk
import torch
from loguru import logger
from .utils import normalization, sitk_utils
from .utils.misc import pad, get_bounding_box_of_mask


class NLSTDataset:

    def __init__(self,
                 scans_dir,
                 masks_dir,
                 labels_path,
                 label,
                 patch_size,
                 hounsfield_units_range=[-600, 1500],
                 transform=None):

        self.scans_paths = list(Path(scans_dir).glob("*.nrrd"))
        self.masks_dir = Path(masks_dir)
        self.labels_df = pd.read_csv(labels_path)
        self.label = label

        self.random_patch_sampler = monai.transforms.RandSpatialCrop(patch_size, random_size=False)
        self.patch_size = patch_size
        self.hounsfield_units_range = hounsfield_units_range
        self.transform = transform

    def __getitem__(self, idx):
        scan_path = self.scans_paths[idx]
        try:
            scan = sitk_utils.load(scan_path)
        except:
            logger.info(f"{scan_path} is corrupted, skipping it.")
            del self.scans_paths[idx]
            return None

        mask_path = self.masks_dir / scan_path.name
        mask = sitk_utils.load(mask_path)

        patient_id = scan_path.stem.replace("_img", "")
        target = self.get_target(patient_id)

        # Originally, mask has a label for each lung, here we combine them into a single label
        mask = sitk.Clamp(mask, upperBound=1)
        # Erosion of the mask helps with getting rid of the lung wall
        mask = sitk.BinaryErode(mask)
        # Mask out everything except lungs
        scan = sitk_utils.apply_mask(scan,
                                     mask,
                                     masking_value=0,
                                     outside_value=self.hounsfield_units_range[0])

        start, end = get_bounding_box_of_mask(mask)
        scan = sitk_utils.slice_image(scan, start=start, end=end)

        tensor = sitk_utils.get_tensor(scan)
        hu_min, hu_max = self.hounsfield_units_range
        tensor = torch.clamp(tensor, hu_min, hu_max)
        tensor = normalization.min_max_normalize(tensor, hu_min, hu_max)
        # Add channel dimension (C,D,H,W)
        tensor = tensor.unsqueeze(0)
        tensor = tensor if self.transform is None else self.transform(tensor)
        tensor = self.random_patch_sampler(tensor)
        # Pad to match the patch size if the resulting patch is smaller
        tensor = pad(tensor, self.patch_size)
        return tensor, target  # torch.tensor().unsqueeze(-1).to(torch.float32)

    def get_target(self, patient_id):
        label_value = self.labels_df[self.labels_df.pid == int(patient_id)][self.label]

        if self.label == "packyear":
            NEVER, LIGHT, MODERATE, HEAVY = 0, 1, 2, 3
            packyear = float(label_value)
            if packyear > 40:
                return HEAVY
            if packyear > 20:
                MODERATE
            if packyear > 1:
                LIGHT
            return NEVER

        raise NotImplementedError

    def __len__(self):
        return len(self.scans_paths)
