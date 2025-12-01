# Video Action Recognition

3D CNN and Video Transformer for action recognition.

## Dataset

**UCF101** - 101 action classes, 13,320 videos. Manual download required:

```bash
cd projects/video_recognition && mkdir -p .datasets
# Download from https://www.crcv.ucf.edu/data/UCF101.php:
# - UCF101.rar
# - UCF101TrainTestSplits-RecognitionTask.zip
unar -o .datasets/ UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip -d .datasets/
```

## Models

- **R3D** - 3D ResNet with spatiotemporal convolutions
- **ViViT** - Video Vision Transformer with tubelet embedding

## Lighter Features Demonstrated

- **Config composition** - `base.yaml` + model config merged via CLI
- **FileWriter + CsvWriter** - save video clips and classification results
- **`_mode_: callable`** - custom collate and writer functions
- **`%` raw references** - shared dataloader settings

## Requirements

```bash
pip install pytorchvideo av
```

## Usage

```bash
cd projects/video_recognition

# macOS Apple Silicon: enable MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# R3D (3D CNN)
uv run --project ../.. lighter fit configs/base.yaml configs/r3d.yaml

# ViViT (Transformer)
lighter fit configs/base.yaml configs/transformer.yaml

# Full training
lighter fit configs/base.yaml configs/r3d.yaml trainer::fast_dev_run=false
```

## References

- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [pytorchvideo](https://pytorchvideo.org/)
