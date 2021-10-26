from pathlib import Path
import SimpleITK as sitk

import sys

sys.path.append("/home/ibrahim/projects/data-pipelines/")

from lungmask import mask
from medimproc.utils.paths import get_files_containing_type


def apply_origin_spacing_direction_from(target, source):
    target.SetOrigin(source.GetOrigin())
    target.SetSpacing(source.GetSpacing())
    target.SetDirection(source.GetDirection())
    return target


def main():
    input_output_dir_pairs = [
        (Path("/mnt/ssd1/NLST/nrrd/T0/"), Path("/mnt/ssd1/NLST/lung_masks/T0/")),
        (Path("/mnt/ssd1/NLST/nrrd/T1/"), Path("/mnt/ssd1/NLST/lung_masks/T1/")),
        (Path("/mnt/ssd1/NLST/nrrd/T2/"), Path("/mnt/ssd1/NLST/lung_masks/T2/")),
    ]
    for pair in input_output_dir_pairs:
        input_dir, output_dir = pair
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in get_files_containing_type(input_dir, type="nrrd"):
            output_path = output_dir / path.name
            if output_path.exists():
                continue

            try:
                scan = sitk.ReadImage(str(path))
            except RuntimeError:
                continue
            try:
                segmentation = mask.apply(scan, batch_size=92)
            except (IndexError, ValueError):
                continue
            segmentation = sitk.GetImageFromArray(segmentation)
            segmentation = apply_origin_spacing_direction_from(segmentation, scan)
            sitk.WriteImage(segmentation, str(output_path), True)


if __name__ == "__main__":
    main()
