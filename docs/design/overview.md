---
title: Architecture & Design
---

# Architecture & Design Overview

Lighter is a configuration-driven deep learning framework that separates experimental setup from code implementation.

## Core Architecture

![Lighter Overview](../assets/images/overview_all.png)
*Figure: Lighter's three-component (bolded) architecture. Config parses YAML definitions, System encapsulates DL components, and Trainer executes training.*

### 1. Config
Transforms YAML experiment definitions into Python objects using MONAI's ConfigParser. One config file = one reproducible experiment.

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

Lighter occupies a unique position in the configuration-driven deep learning landscape:

| Feature | **Lighter** | **[Ludwig](https://github.com/ludwig-ai/ludwig)** | **[Quadra](https://github.com/orobix/quadra)** | **[GaNDLF](https://github.com/mlcommons/GaNDLF)** |
|---------|------------|-----------|------------|------------|
| **Lines of Code** | ~1,000 | ~100,000 | ~10,000 | ~50,000 |
| **Abstraction Level** | Medium | High | High | High |
| **Task Coverage** | Any PyTorch task | Multi-modal | Vision | Medical |
| **Custom Code** | Seamless | Limited | Moderate | Limited |
| **Flexibility** | Maximum (adapters) | Low | Moderate | Domain-specific |

Each framework serves different needs. Lighter's strength: minimal abstraction for maximum control.

## Next Steps

- Deep dive into [the Adapter Pattern](adapters.md)
- Understand [Design Philosophy](philosophy.md)
- Get started with the [Zero to Hero tutorial](../tutorials/zero_to_hero.md)
