from pathlib import Path
import SimpleITK as sitk
from lungmask import mask

import sys

sys.path.append("../../../../data-pipelines")

from medimproc.utils.paths import get_files_containing_type


def apply_origin_spacing_direction_from(target, source):
    target.SetOrigin(source.GetOrigin())
    target.SetSpacing(source.GetSpacing())
    target.SetDirection(source.GetDirection())
    return target


def main():
    input_dir = Path("/mnt/data1/NLST/nrrd/T2/")
    output_dir = Path("/mnt/ibrahim/NLST/nrrd/lung_masks/T2/")
    output_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    for path in get_files_containing_type(input_dir, type="nrrd"):
        output_path = output_dir / path.name
        if output_path.exists():
            continue

        try:
            scan = sitk.ReadImage(str(path))
        except:
            i += 1
            print(i, path)
            continue
        segmentation = mask.apply(scan, batch_size=32)
        segmentation = sitk.GetImageFromArray(segmentation)
        segmentation = apply_origin_spacing_direction_from(segmentation, scan)
        sitk.WriteImage(segmentation, str(output_path), True)


if __name__ == "__main__":
    main()
