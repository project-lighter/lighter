# Example projects

After [installing the current source pair](../docs/guides/compatibility.md#install-the-current-source-pair), start with [Compare and Continue](experiment_comparison/README.md): inspect a recipe, compare two learning rates on fixed CIFAR-10 populations, evaluate the selected checkpoint, export identified predictions and continue with restored optimizer state. The matched native Lightning implementation makes the scientific behavior and the Lighter convenience visible side by side. The recorded CPU results are specific to the declared subset, seeds and software profile.

For the complete first-use command sequence, follow the [quick start](../docs/quickstart.md). Its installation check uses [tabular_regression](tabular_regression/README.md). It preserves a small known-answer fit/evaluate/export/continue diagnostic. It is not a second public research study.

## Status of earlier projects

These sources remain available for reference and attribution. They are not all maintained or qualified research protocols. A successful import, configuration inspection or another project's passing check does not certify their scientific behavior.

| Project | Disposition and requirement before recommendation |
| --- | --- |
| [tabular_regression](tabular_regression/README.md) | Retained download-free diagnostic/reference; historical qualification retained. |
| [cifar10](cifar10/README.md) | Legacy classifier; Compare and Continue replaces the recommended entry. |
| [huggingface_llm](huggingface_llm/README.md) | Legacy text integration; requires a held-out, variable-length text workflow and actual model profile. |
| [self_supervised](self_supervised/README.md) | Archived from recommended gallery; reopen for a representation-transfer question with controls. |
| [lora](lora/README.md) | Archived from recommended gallery; actual PEFT training/restoration must establish adapter behavior. |
| [vision_language](vision_language/README.md) | Specialist reference; fixed-gallery multi-positive retrieval needs explicit identities and controls. |
| [video_recognition](video_recognition/README.md) | Specialist reference; qualify real decoding, layout and clip aggregation for one model. |
| [medical_segmentation](medical_segmentation/README.md) | Specialist reference; qualify spatial geometry and exported volumes on a declared MONAI/ITK profile. |
| [eeg](eeg/README.md) | Specialist reference; establish participant/window semantics, worker RNG and actual export. |

## Build your own experiment

Copy the public project's structure, then replace its scientific task and data policy explicitly. Keep a `__lighter__.py` marker and `__init__.py` for `project.*` imports. Scientific steps remain ordinary Python; configuration composes objects, values and dependencies. Native LightningModules retain their own optimization and lifecycle hooks. `@` shares an object; `%` copies a definition. A new task needs its own independent scientific checks even when its recipe looks similar.

Choose [native ownership](../docs/guides/lightning-module.md) or [LighterModule](../docs/guides/lighter-module.md) based on the task. See [custom code](../docs/guides/custom-code.md), [experiment records](../docs/guides/experiment-records.md) and [compatibility](../docs/guides/compatibility.md).
