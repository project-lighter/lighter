---
title: 'Lighter: Configuration-Driven Deep Learning'
tags:
  - Python
  - PyTorch
  - deep learning
  - configuration
  - framework
authors:
  - name: Ibrahim Hadzic
    orcid: 0000-0002-8397-5940
    corresponding: true
    affiliation: "1, 2"
  - name: Suraj Pai
    orcid: 0000-0001-8043-2230
    affiliation: "2, 3, 4"
  - name: Keno Bressem
    affiliation: "5, 6"
    orcid: 0000-0001-9249-8624
  - name: Borek Foldyna
    affiliation: 1
    orcid: 0000-0002-2466-4827
  - given-names: Hugo
    dropping-particle: JWL
    surname: Aerts
    affiliation: "2, 3, 4"
    orcid: 0000-0002-2122-2003

affiliations:
 - name: Cardiovascular Imaging Research Center, Massachusetts General Hospital, Harvard Medical School, United States of America
   index: 1
 - name: Radiology and Nuclear Medicine, CARIM & GROW, Maastricht University, The Netherlands
   index: 2
 - name: Artificial Intelligence in Medicine (AIM) Program, Mass General Brigham, Harvard Medical School, Harvard Institutes of Medicine, United States of America
   index: 3
 - name: Department of Radiation Oncology, Brigham and Women’s Hospital, Dana-Farber Cancer Institute, Harvard Medical School, United States of America
   index: 4
 - name: Technical University of Munich, School of Medicine and Health, Klinikum rechts der Isar, TUM University Hospital, Germany
   index: 5
 - name: Department of Cardiovascular Radiology and Nuclear Medicine, Technical University of Munich, School of Medicine and Health, German Heart Center, TUM University Hospital, Germany
   index: 6
date: 24 February 2025
bibliography: paper.bib

---

# Summary

Lighter is a configuration-driven deep learning (DL) [framework](https://github.com/project-lighter/lighter) that separates experimental setup from code implementation. Models, datasets, and other components are defined through structured configuration files (configs). Configs serve as snapshots of the experiments, enhancing reproducibility while eliminating unstructured and repetitive scripts. Lighter uses (i) PyTorch Lightning [@Falcon_PyTorch_Lightning_2019] to implement a task-agnostic DL logic, and (ii) [MONAI Bundle configuration](https://docs.monai.io/en/stable/config_syntax.html#) [@Cardoso_MONAI_An_open-source_2022] to manage experiments using YAML configs.

# Statement of Need

Lighter addresses several challenges in DL experimentation:

1. **Repetitive and Error-Prone Setups**: DL typically involves significant boilerplate code for training loops, data loading, and metric calculations. The numerous hyperparameters and components across experiments can easily become complex and error-prone. Lighter abstracts these repetitive tasks and uses centralized configs for a clear, manageable experimental setup, reducing tedium and potential for errors.

2. **Reproducibility and Collaboration**: Inconsistent or complex codebases hinder collaboration and experiment reproduction. Lighter's self-documenting configs offer clear, structured snapshots of each experiment. This greatly improves reproducibility and simplifies how teams share and reuse setups.

3. **Pace of Research Iteration**: The cumulative effect of these challenges inherently slows down the research cycle. Lighter streamlines the entire experimental process, allowing researchers to focus on core hypotheses and iterate on ideas  efficiently.

# State of the Field

Config-driven frameworks like Ludwig [@Ludwig], Quadra [@Quadra], and GaNDLF [@Gandlf] offer high level of abstraction by providing predefined structures and pipelines. While this approach simplifies usage, it limits flexibility to modify the pipeline or extend components, often requiring direct source code changes.
Lighter takes a different approach by providing medium-level abstraction. It implements a flexible pipeline that maintains direct compatibility with standard PyTorch components (models, datasets, optimizers). The pipeline itself is modifiable to any task via [adapters](#adapters), while custom code is [importable via config](#project-specific-modules) without source code modifications.

# Design

Lighter is built upon three fundamental components (\autoref{fig:overview_all}):

1.  **`Config`**: serves as the primary interface for interacting with Lighter. It parses and validates YAML configs that define all components, creating a self-documenting record of each experiment.

2.  **`System`**: encapsulates the components (model, optimizer, scheduler, loss function, metrics, and dataloaders) and connects them into a pipeline that can be customized through [adapters](#adapters) (\autoref{fig:overview_system}).

3. **`Trainer`**:  PyTorch Lightning's `Trainer` handles aspects like distributed or mixed-precision training and checkpoint management. Lighter uses it to execute the protocol defined by the `System`.

![**Lighter Overview.** `Config` leverages MONAI's `ConfigParser` for parsing the user-defined YAML configs, and its features are used by Runner to instantiate the `System` and `Trainer`. `Trainer` is used directly from PyTorch Lightning, whereas `System` inherits from `LightningModule`, ensuring its compatibility with `Trainer` while implementing a logic generalizable to any task or type of data. Finally, `Runner` runs the paired `Trainer` and `System` for a particular stage (e.g., fit or test).\label{fig:overview_all}](overview_all.png)

![**Flowchart of the `lighter.System`.** A `batch` from the `DataLoader` is processed by `BatchAdapter` to extract `input`, `target` (optional), and `identifier` (optional). The `Model` generates `pred` (predictions) from the `input`. `CriterionAdapter` and `MetricsAdapter` compute loss and metrics, respectively, by applying optional transformations and routing arguments for the loss and metric functions. Results, including loss, metrics, and other data prepared for logging by the `LoggingAdapter` are returned to the `Trainer`.\label{fig:overview_system}](overview_system.png)

## Adaptability Through Modular Design

### Adapters

If we consider all possible DL tasks, we will find it challenging to implement a single pipeline that supports all. Instead, frameworks often implement per-task pipelines (e.g., segmentation, classification, etc.). By contrast, Lighter implements a unified pipeline modifiable via *adapter classes*. In software design, *adapter design pattern* enables components with incompatible interfaces to work together by *bridging* them using an adapter class. In Lighter, these bridges (\autoref{fig:overview_system}) specify how components should interact across data types and tasks. For example, a model's output will differ based on the task (e.g., segmentation, regression), and the adapter will specify how to pass them on to the next component (e.g., criterion or metrics). This design allows Lighter to handle any task without requiring changes to the source code.

```yaml
# Example of an adapter transforming and routing data to the loss function
adapters:
    train:
        criterion:
            _target_: lighter.adapters.CriterionAdapter
            pred_transforms:   # Apply sigmoid activation to predictions
                _target_: torch.sigmoid
            pred_argument: 0   # Pass 'pred' to criterion's first arg
            target_argument: 1 # Pass 'target' to criterion's second arg
```

### Project-specific modules

Using custom components does not require modifying the framework. Instead, they can be defined within a *project folder* like:

```
joss_project
├── __init__.py
└── models/
    ├── __init__.py
    └── mlp.py
```

By specifying the project path in the config, it is imported as a module whose components can be referenced in the config:

```yaml
project: /path/to/joss_project  # Path to the directory above
system:
    model:
        _target_: project.models.mlp.MLP  # Reference to the custom model
        input_size: 784
        num_classes: 10
```

# Research Contributions That Use Lighter

- Foundation model for cancer imaging biomarkers [@Pai2024]
- Vision Foundation Models for Computed Tomography [@Pai2025]

# Acknowledgments

We thank John Zielke for the adapter design pattern idea. We thank Wenqi Li, Nic Ma, Yun Liu, and Eric Kerfoot for their continuous support with MONAI Bundle.

# References
