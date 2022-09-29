from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from .utils import resample_image_to_spacing, slice_image

class CIDataset(Dataset):
    """
    add documentation on how this dataset works

    Args:
        add docstrings for the parameters
    """

    def __init__(self, path, label=None, radius=25, orient=False, resample_spacing=None, transform=None):
        super(CIDataset, self).__init__()
        self._path = Path(path)

        self.radius = radius
        self.orient = orient
        self.resample_spacing = resample_spacing
        self.label = label
        self.transform = transform

        self.annotations = pd.read_csv(self._path)
        self._num_samples = len(self.annotations)  # set the length of the dataset

    def get_image_paths(self):
        return self.annotations["image_path"].values

    def get_labels(self):
        """ "
        Function to get labels for when they are available in the dataset
        For example this is to be used for the medical image dataset with labels
        available in the annotations
        """

        labels = self.annotations[self.label].values
        assert not np.any(labels == -1), "All labels must be specified"
        return labels

    def __len__(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """

        # Get a row from the CSV file
        row = self.annotations.iloc[idx]
        image_path = row["image_path"]
        assert self.label in row, "Label column not found in dataframe"
        target = int(row[self.label])

        image = sitk.ReadImage(str(image_path))
        image = resample_image_to_spacing(image, self.resample_spacing, -1024) \
                if self.resample_spacing is not None else image

        centroid = (row["coordX"], row["coordY"], row["coordZ"])
        centroid = image.TransformPhysicalPointToContinuousIndex(centroid)
        centroid = [int(d) for d in centroid]

        # Orient all images to LPI orientation
        image = sitk.DICOMOrient(image, "LPI") if self.orient else image

        # Extract nodule with a specified radius around centroid
        nodule_image = slice_image(
            image,
            [(centroid[idx] - self.radius) for idx in range(3)],
            [(centroid[idx] + self.radius) for idx in range(3)],
        )
        array = sitk.GetArrayFromImage(nodule_image)
        tensor = array if self.transform is None else self.transform(array)
        return tensor, target