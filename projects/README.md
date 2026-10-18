# Example Projects

Example projects demonstrating Lighter across domains. Run each one from inside its folder, e.g.
`cd cifar10 && lighter fit configs/example.yaml`: paths in the configs, such as `./.datasets/`, are relative to it.

| Project | Domain | Extra dependencies | Tested in CI |
|---------|--------|--------------------|--------------|
| [tabular_regression](./tabular_regression/) | Regression on synthetic data, and the docs' recipes | - | Every stage and recipe, on every pull request |
| [cifar10](./cifar10/) | Image classification | torchvision | Every stage on random images, on every pull request; on CIFAR-10 on `main` |
| [eeg](./eeg/) | EEG analysis | braindecode, eegdash, mne | No |
| [huggingface_llm](./huggingface_llm/) | Sentiment classification | transformers, datasets | No |
| [lora](./lora/) | Parameter-efficient fine-tuning (LoRA) | peft, torchvision | Its data split only |
| [medical_segmentation](./medical_segmentation/) | Medical imaging | monai, itk | No |
| [self_supervised](./self_supervised/) | Self-supervised computer vision | lightly | No |
| [video_recognition](./video_recognition/) | Video understanding | pytorchvideo, av | No |
| [vision_language](./vision_language/) | Vision-language | transformers | No |

The projects that CI does not run need their extra dependencies and a dataset download. Nothing checks them
against the current Lighter, so they may need changes to run.
