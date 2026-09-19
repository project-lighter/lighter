# Example projects

Choose an example by the decision you need to make.

| Example | Purpose | Prerequisites |
|---|---|---|
| [Download-free diagnostic](../quickstart.md) | Check fit/evaluation/export/continuation and local records | Current source pair, CPU; generated data |
| [Compare and Continue](compare-and-continue.md) | Compare learning rates, select on validation, evaluate a checkpoint, inspect restored state | Current pair, prepared CIFAR-10 subset, declared CPU budget |
| Earlier integrations in the example checkout’s `projects/` directory | Explore domain-specific reference code | Each project's scientific and dependency checks |

## What to learn from the walkthrough

Compare and Continue keeps scientific steps and data policy in ordinary Python. YAML composes the task, native Trainer and callbacks; one small overlay changes the requested LR. Its native Lightning counterpart makes the same scientific behavior visible without recreating Lighter's record service.

The workflow distinguishes validation-selected versus last checkpoints, immutable evaluation inputs, identified predictions and full optimizer restoration. Its README shows ordinary commands first and optional maintainer verification afterward. All recorded conditions and limits remain visible.

## Earlier integrations

The old CIFAR classifier is superseded for onboarding. Text, LoRA and self-supervision remain legacy references; vision-language, video, segmentation and EEG are specialist illustrations awaiting their own declared scientific/dependency qualification. A passing import or another example's result does not make them universally ready research protocols.

See `projects/README.md` in the pinned example checkout for each disposition. These examples demonstrate possible domains; they do not make medical imaging the organizing scope of Lighter.

## Create your own project

Begin with [custom code](../guides/custom-code.md), select [native](../guides/lightning-module.md) or [managed](../guides/lighter-module.md) ownership, then define your scientific question, populations, update policy and outputs. Reuse a working structure without assuming its loss, metric or recovery semantics apply to a different task.

Keep the [research checks](../guides/best-practices.md) proportionate to the new behavior. No control factories, evaluation platform or agent-specific authoring interface is required.
