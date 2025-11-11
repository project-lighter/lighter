---
title: Architecture & Design
---

# Architecture & Design Overview

Lighter is a configuration-driven deep learning framework that separates experimental setup from code implementation.

## Core Architecture

![Lighter Overview](../assets/images/overview_all.png)
*Figure: Lighter's three-component (bolded) architecture. Config parses YAML definitions, System encapsulates DL components, and Trainer executes training.*

### 1. Config
Transforms YAML experiment definitions into Python objects using Sparkwheel. One config file = one reproducible experiment.

[→ Configuration guide](../how-to/configure.md)

### 2. System
Orchestrates your deep learning pipeline—model, optimizer, loss, metrics, data. Extends PyTorch Lightning's LightningModule.

### 3. Trainer
PyTorch Lightning's Trainer executes experiments with multi-GPU, mixed precision, gradient accumulation, and checkpointing.

[→ Running experiments](../how-to/run.md)

## The Adapter Pattern

Adapters make Lighter task-agnostic by handling data format differences between components.

[→ Learn more about adapters](adapters.md)

## Design Philosophy

Lighter follows four core principles: **Configuration over Code**, **Composition over Inheritance**, **Convention over Configuration**, and **Separation of Concerns**.

[→ Understand the philosophy](philosophy.md)

## Framework Comparison

Lighter's goal is to brings reproducibility and structure, while keeping you in full control of your code. This is different from other configuration-driven frameworks that provide higher-level abstractions.

| Feature | **Lighter** | **[Ludwig](https://github.com/ludwig-ai/ludwig)** | **[Quadra](https://github.com/orobix/quadra)** | **[GaNDLF](https://github.com/mlcommons/GaNDLF)** |
|---|---|---|---|---|
| **Primary Focus** | Config-driven, task-agnostic DL | Config-driven, multi-task DL | Config-driven computer vision | Config-driven medical imaging |
| **Configuration** | YAML (Sparkwheel) | YAML (Custom) | YAML (Hydra) | YAML (Custom) |
| **Abstraction** | Medium. Extends PyTorch Lightning, expects standard PyTorch components. | High. Provides pre-built flows for various tasks. | High. Pre-defined structures for computer vision. | High. Pre-defined structures for medical imaging. |
| **Flexibility** | High. New components are added via project module. | Medium. Adding new components requires code editing. | Low. Adding new components requires code editing. | Low. Adding new components requires code editing. |
| **Use Case** | Organized experimentation | Production-level applications | Traditional computer vision | Established medical imaging methods |

Lighter is the tool for you if you like PyTorch's flexibility but want to manage your experiments in a structured and reproducible way.

## Next Steps

- Deep dive into [the Adapter Pattern](adapters.md)
- Understand [Design Philosophy](philosophy.md)
- Get started with the [Zero to Hero tutorial](../tutorials/zero_to_hero.md)
