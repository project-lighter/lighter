# Medical Image Segmentation with MONAI

3D CT spleen segmentation using MONAI.

## Dataset

**Medical Segmentation Decathlon - Task09_Spleen**
- 41 training + 20 validation CT volumes
- Auto-downloaded (~1.5 GB)

## Architecture

**MONAI UNet** - 3D U-Net with residual connections and instance normalization.

## Lighter Features Demonstrated

- **MONAI integration** - DecathlonDataset, transforms, UNet
- **Sliding window inference** - process large 3D volumes in patches
- **Mixed precision** - 16-bit training
- **FileWriter callback** - saves segmentation masks as .seg.nrrd
- **`_mode_: callable`** - for custom writer function

## Requirements

```bash
pip install monai
```

## Usage

```bash
cd projects/medical_segmentation

# Quick test
uv run --project ../.. lighter fit configs/spleen.yaml

# Full training
lighter fit configs/spleen.yaml trainer::fast_dev_run=false

# Multi-GPU (recommended for 3D)
lighter fit configs/spleen.yaml trainer::devices=2 trainer::strategy=ddp
```

## References

- [MONAI Documentation](https://docs.monai.io/)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
