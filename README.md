
<br/>
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
<br/>
<br/>
<div align="center">

[![build](https://github.com/project-lighter/lighter/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/project-lighter/lighter/actions/workflows/build.yml) ![Coverage](./assets/images/coverage.svg) [![GitHub license](https://img.shields.io/github/license/project-lighter/lighter)](https://github.com/project-lighter/lighter/blob/main/LICENSE)

<a href="https://discord.gg/zJcnp6KrUp">
  <img src="https://discord.com/api/guilds/1252251284908539965/widget.png?style=banner2" alt="Lighter Discord Server"/>
</a>
</div>


&nbsp;


Focus on your deep learning experiments and forget about (re)writing code. `lighter` is:
 1. **Task-agnostic**

    Whether you’re working on classification, segmentation, or self-supervised learning, `lighter` provides generalized training logic that you can use out-of-the-box.

 2. **Configuration-based**

    Easily define, track, and reproduce experiments with `lighter`’s configuration-driven approach, keeping all your hyperparameters organized.

 3. **Customizable**

    Seamlessly integrate your custom models, datasets, or transformations into `lighter`’s flexible framework.

&nbsp;

`lighter` stands on the shoulder of these two giants:
 - [MONAI Bundle](https://docs.monai.io/en/stable/bundle_intro.html) - Configuration system. Similar to [Hydra](https://github.com/facebookresearch/hydra), but with additional features.
 - [PyTorch Lightning](https://github.com/Lightning-AI/lightning) - Our [`System`](https://project-lighter.github.io/lighter/reference/system/) is based on the PyTorch Lightning [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) and implements all the necessary training logic for you. Couple it with the PyTorch Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) and you're good to go.

<br/>
<div align="center">Simply put, <code>lighter = config(trainer + system)</code>  😇</div>
<br/>

## 📖 Getting Started
<div align="center">
<p style="text-align: center;">
  📚 <a href="https://project-lighter.github.io/lighter/"> Documentation</a>&nbsp;&nbsp;&nbsp;
  🎥 <a href="https://www.youtube.com/channel/UCef1oTpv2QEBrD2pZtrdk1Q">YouTube Channel</a>&nbsp;&nbsp;&nbsp;
  👾 <a href="https://discord.gg/zJcnp6KrUp">Discord Server</a>
</p>
</div>

<b>Install:</b>
<pre><code>pip install project-lighter</code></pre>
<details>
<summary><b>Pre-release (up-to-date with the main branch):</b></summary>
<pre><code>pip install project-lighter --pre</code></pre>
</details>

<details>
<summary><b>For development:</b></summary>
<pre><code>make setup
make install             # Install lighter via uv
make pre-commit-install  # Set up the pre-commit hook for code formatting
</details>
<br/>


## 💡 Projects
Projects that use `lighter`:

| Project | Description |
| --- | --- |
| [Foundation Models for Quantitative Imaging Biomarker Discovery in Cancer Imaging](https://aim.hms.harvard.edu/foundation-cancer-image-biomarker) | A foundation model for lesions on CT scans that can be applied to down-stream tasks related to tumor radiomics, nodule classification, etc. |

<br/>

## Cite
<pre><code>@software{lighter,
author       = {Ibrahim Hadzic and
                Suraj Pai and
                Keno Bressem and
                Hugo Aerts},
title        = {Lighter},
publisher    = {Zenodo},
doi          = {10.5281/zenodo.8007711},
url          = {https://doi.org/10.5281/zenodo.8007711}
}</code></pre>
</div>
