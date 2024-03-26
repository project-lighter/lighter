<div align="center">
<picture>
  <!-- old code that allows different pics for light/dark mode -->
  <!--
  <source media="(prefers-color-scheme: dark)" srcset="./assets/images/lighter_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./assets/images/lighter_light.png">
   -->
  <img align="center" alt="Lighter logo" src="./assets/images/lighter.png">
</picture>
</div>
<br/>
<div align="center">

 [![build](https://github.com/project-lighter/lighter/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/project-lighter/lighter/actions/workflows/build.yml) ![Coverage](./assets/images/coverage.svg) [![GitHub license](https://img.shields.io/github/license/project-lighter/lighter)](https://github.com/project-lighter/lighter/blob/main/LICENSE)
</div>


With `lighter`, focus on your deep learning experiments and forget about boilerplate through:
 1. **Task-agnostic** training logic already implemented for you (classification, segmentation, self-supervised, etc.)
 2. **Configuration-based** approach that will ensure that you can always reproduce your experiments and know what hyperparameters you used.
 3. Extremely **simple integration of custom** models, datasets, transforms, or any other components to your experiments.

&nbsp;

`lighter` stands on the shoulder of these two giants:
 - [MONAI Bundle](https://docs.monai.io/en/stable/bundle_intro.html) - Configuration system. Similar to [Hydra](https://github.com/facebookresearch/hydra), but with additional features.
 - [PyTorch Lightning](https://github.com/Lightning-AI/lightning) - Our [`LighterSystem`](https://project-lighter.github.io/lighter/reference/system/) is based on the PyTorch Lightning [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) and implements all the necessary training logic for you. Couple it with the PyTorch Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) and you're good to go.
 
Simply put, `lighter = config(trainer + system)` 😇


## 📖 Usage

- [📚 Documentation](https://project-lighter.github.io/lighter/)
- [🎥 YouTube Channel](https://www.youtube.com/channel/UCef1oTpv2QEBrD2pZtrdk1Q)

## 🚀 Install

Current release:
```bash
pip install project-lighter
```

Pre-release (up-to-date with the main branch):
```bash
pip install project-lighter --pre
```

For development:
```bash
make setup
make install             # Install lighter via Poetry
make pre-commit-install  # Set up the pre-commit hook for code formatting
poetry shell             # Once installed, activate the poetry shell
```

## 💡 Projects
Projects that use `lighter`:

| Project | Description |
| --- | --- |
| [Foundation Models for Quantitative Imaging Biomarker Discovery in Cancer Imaging](https://aim.hms.harvard.edu/foundation-cancer-image-biomarker) | A foundation model for lesions on CT scans that can be applied to down-stream tasks related to tumor radiomics, nodule classification, etc. |


## 📄 Cite:

If you find `lighter` useful in your research or project, please consider citing it:

```bibtex
@software{lighter,
  author       = {Ibrahim Hadzic and
                  Suraj Pai and
                  Keno Bressem and
                  Hugo Aerts},
  title        = {Lighter},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8007711},
  url          = {https://doi.org/10.5281/zenodo.8007711}
}
```

We appreciate your support!
