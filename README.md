
<br/>
<div align="center">
<picture>
  <img align="center" alt="Lighter logo" src="https://raw.githubusercontent.com/project-lighter/lighter/main/assets/images/lighter.png" width="80%">
</picture>
</div>
<br/>
<br/>

<div align="center">

[![build](https://github.com/project-lighter/lighter/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/project-lighter/lighter/actions/workflows/ci.yml) ![Coverage](./assets/images/coverage.svg) [![GitHub license](https://img.shields.io/github/license/project-lighter/lighter)](https://github.com/project-lighter/lighter/blob/main/LICENSE) [![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/zJcnp6KrUp?style=flat)](https://discord.gg/zJcnp6KrUp)

</div>

<br/>
<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/project-lighter/lighter/main/assets/images/features_dark.png" width="85%">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/project-lighter/lighter/main/assets/images/features_light.png" width="85%">
        <img alt="Features" src="https://raw.githubusercontent.com/project-lighter/lighter/main/assets/images/features_light.png" width="85%">
    </picture>
</div>
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
<pre><code>pip install lighter</code></pre>
<details>

<summary><b>Install pre-release</b></summary>
<p>Get the latest features and fixes from the main branch.</p>
<pre>
<code>pip install lighter --pre</code>
</pre>
</details>

<details>
<summary><b>Development:</b></summary>
<p>To contribute to the project, clone the repository and run the following commands. Also, refer to the <a href="CONTRIBUTING.md">contributing guide</a>.</p>
<pre>
<code>make setup
make install             # Install lighter via uv
make pre-commit-install  # Set up the pre-commit hook for code formatting</code>
</pre>
</details>
<br/>


## 💡 Projects
Projects that use `lighter`:

| Project | Description |
| --- | --- |
| [Foundation Models for Quantitative Imaging Biomarker Discovery in Cancer Imaging](https://aim.hms.harvard.edu/foundation-cancer-image-biomarker) | A foundation model for lesions on CT scans that can be applied to down-stream tasks related to tumor radiomics, nodule classification, etc. |
| [Vision Foundation Models for Computed Tomography](https://arxiv.org/abs/2501.09001) | A large-scale 3D foundation model for CT scans demonstrating superior performance in segmentation, triage, retrieval, and semantic understanding |

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
