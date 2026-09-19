# Medical Image Segmentation with MONAI

> **Status: Specialist reference.** This integration is outside the maintained onboarding path. Reopening it requires a declared MONAI/ITK profile and independent spatial geometry, phantom and exported-volume checks. Lighter is general purpose; this is a specialist validation boundary, not a restriction on medical applications. Start with [Compare and Continue](../experiment_comparison/README.md), or use the [download-free diagnostic](../tabular_regression/README.md). See the [project index](../README.md) for qualification scope.

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

## Usage

```bash
pip install lighter monai itk
cd projects/medical_segmentation

# Train
lighter fit configs/spleen.yaml

# Quick test
lighter fit configs/spleen.yaml trainer::fast_dev_run=true

# Multi-GPU (recommended for 3D)
lighter fit configs/spleen.yaml trainer::devices=2 trainer::strategy=ddp
```

## References

- [MONAI Documentation](https://docs.monai.io/)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
