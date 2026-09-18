---
title: Example Projects
---

# Example projects

The recommended research walkthrough is [Compare and Continue](https://github.com/project-lighter/lighter/tree/main/projects/experiment_comparison). It asks a concrete question about two learning rates on fixed CIFAR-10 populations, then follows the selected checkpoint through evaluation, identified prediction export and checkpoint continuation. An equivalent native Lightning implementation provides a fair comparison. The README records every included condition and the limits of its small CPU experiment.

Use the [download-free diagnostic](../quickstart.md) to check an installation with a known synthetic regression problem. Its role is a small reference workflow; the public comparison is the next step for research use.

## What to learn from the walkthrough

- Keep scientific steps and data policy in ordinary Python, with native Lightning owning execution.
- Compose the same task, data, Trainer and callbacks in a readable recipe; change one scalar with an overlay.
- Inspect definitions before constructing objects. Distinguish requested settings from observed and restored optimizer state.
- Select on validation data, then evaluate a concrete immutable checkpoint against exact final-population IDs.
- Continue from the last full-state checkpoint into a new attempt. Preserve the original selected model separately.

## Earlier integrations

The [repository project index](https://github.com/project-lighter/lighter/blob/main/projects/README.md) records the disposition of all earlier projects. The old CIFAR classifier is superseded for onboarding; text, LoRA and self-supervision examples are legacy references; vision-language, video, segmentation and EEG are specialist illustrations awaiting their own declared scientific and dependency qualification. Their sources remain available, with status notices and migration pointers. They are not a set of universally ready-to-run research protocols.

## Create your own project

Use the public comparison as a structural example, then define your own question, data populations and correctness controls. Place empty `__lighter__.py` and `__init__.py` files in the project so recipes can import `project.*`. Native Lightning modules can keep their custom optimization and hooks; ordinary Lighter modules can use managed optimizer construction. Neither path requires LightningCLI or author-written factories.

See [custom code](../guides/custom-code.md), [LighterModule](../guides/lighter-module.md), [native LightningModule](../guides/lightning-module.md) and [experiment records](../guides/experiment-records.md).
