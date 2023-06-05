<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/images/lighter_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./assets/images/lighter_light.png">
  <img align="center" alt="Lighter logo" src="h/assets/images/lighter_dark.png">
</picture>
</div>
<br/>
<div align="center">

 [![build](https://github.com/project-lighter/lighter/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/project-lighter/lighter/actions/workflows/build.yml) ![Coverage](./assets/images/coverage.svg) [![GitHub license](https://img.shields.io/github/license/project-lighter/lighter)](https://github.com/project-lighter/lighter/blob/main/LICENSE)
</div>


Welcome to `lighter`, an elegant and powerful wrapper for [PyTorch Lightning](https://github.com/Lightning-AI/lightning) that simplifies the way you build and manage your deep learning experiments. Unleash your model's potential through a unified, **configuration-based** approach that streamlines the experimentation process, empowering both beginners and experts in the field.


## 🚀 Install

Current release:
````
pip install project-lighter
````

Pre-release (up-to-date with the main branch):
````
pip install project-lighter --pre
````

For development:
````
make setup
make install             # Install lighter via Poetry
make pre-commit-install  # Set up the pre-commit hook for code formatting
poetry shell             # Once installed, activate the poetry shell
````

## 📖 Usage

- [Documentation]()
- [Video Tutorials]()

## 💡 Projects
List of projects that use `lighter`:

| Project | Description |
| --- | --- |
| [FMCBI]() | |


## 📄 Cite:

If you find `lighter` useful in your research or project, please consider citing it. Here's an example BibTeX citation entry:

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
