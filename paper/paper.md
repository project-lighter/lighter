---
title: 'Lighter: Configuration-Driven Deep Learning'
tags:
  - Python
  - PyTorch
  - deep learning
  - configuration
authors:
  - name: Ibrahim Hadzic
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: "1, 2"
  - name: Suraj Pai
    orcid: 0000-0000-0000-0000
    affiliation: "2, 3, 4"
  - name: Keno Bressem
    affiliation: "5, 6"
  - name: Borek Foldyna
    affiliation: 1
  - given-names: Hugo
    surname: Aerts
    affiliation: "2, 3, 4"
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
 - name: Department of Cardiovascular Radiology and Nuclear Medicine, Technical University of Munich, School of Medicine and Health, German Heart Center, TUM University Hospital, Germay
   index: 6
date: 22 February 2025
bibliography: paper.bib

---

# Summary

Lighter is a Python [framework](https://github.com/project-lighter/lighter) designed to streamline deep learning experimentation by leveraging PyTorch Lightning [@Falcon_PyTorch_Lightning_2019] and the [MONAI Bundle configuration](https://docs.monai.io/en/stable/config_syntax.html#) [@Cardoso_MONAI_An_open-source_2022]. It employs intuitive YAML configuration files to comprehensively define experiments, encompassing models, optimizers, data loaders, metrics, and more. This approach separates experiment-specific components from the core training and inference logic, effectively eliminating boilerplate code. By ensuring reproducibility through self-documenting configurations, Lighter facilitates collaboration and accelerates research cycles. Its adaptable adapter system and support for project-specific modules empower users to tackle a wide range of deep learning tasks without altering the core framework. By handling the complexities of training infrastructure, Lighter enables researchers to concentrate on innovation and swiftly iterate on their ideas.

# Statement of Need

Lighter is designed to address several key challenges in deep learning experimentation:

1.  **Boilerplate Code:** The repetitive nature of writing training loops, data loading, metric calculations, and experiment setups can vary significantly between projects. *Lighter abstracts these repetitive tasks, exposing only the components that differ across projects.*

2.  **Experiment Management:** Handling numerous hyperparameters and configurations across various experiments can become cumbersome and error-prone. *Lighter offers organized configuration through YAML files, providing a **centralized record of all experiment parameters**.*

3.  **Reproducibility:** Reproducing experiments from different implementations can be challenging. *Lighter's **self-contained configuration files** serve as comprehensive documentation, facilitating the exact recreation of experimental setups.*

4.  **Collaboration:** Collaborating on experiments often requires understanding complex codebases. *Lighter enhances collaboration by using standardized configurations, making it easier to share and reuse experiment setups within and across research teams.*

5.  **Slowed Iteration:** The cumulative effect of these challenges can significantly slow down the research iteration cycle. *Lighter accelerates iteration by streamlining the experiment setup process, allowing researchers to focus on core experiment choices without being bogged down by infrastructure concerns.*


# Design

Lighter's design is centered around three core components: `Config`, `System`, and `Trainer`:

1.  **`Config`**: This component is responsible for parsing and validating YAML configuration files that define the experiment setup. The `System` and `Trainer` are defined within this configuration.

2.  **`System`**: Acting as the central orchestrator, the `System` class manages the experiment's components and the data flow between them. It encapsulates the model, optimizer, scheduler, criterion, metrics, dataloaders, and adapters, implementing their behavior during training, validation, testing, and prediction.

3. **`Trainer`**: Leveraging PyTorch Lightning's `Trainer` class, this component manages the training process, handling tasks such as distributed training, mixed precision, and checkpointing. Lighter utilizes the `Trainer` to execute the setup defined by the `System`.


![**Lighter Overview.** Lighter revolves around three main components -- `Trainer,` `System` and `Config`, which contains the definition for the former two. `Config` leverages MONAI's `ConfigParser` for parsing the user-defined YAML configuration files, and its features are used by Runner to instantiate the `System` and `Trainer`. `Trainer` is used directly from PyTorch Lightning, whereas `System` inherits from `LightningModule`, ensuring its compatibility with `Trainer` while implementing a logic generalizeable to any task or type of data. Finally, `Runner` runs the paired `Trainer` and `System` for a particular stage (e.g., fit or test).](overview_all.png)

<!-- ## Config -->

## System

Lighter's `System` encapsulates components that typically vary across experiments—such as the model, optimizer, scheduler, criterion, metrics, and dataloaders—and defines a general flow of data between these components. Crucially, this flow is adjustable using another component within `System`, [adapters](#adapters), which allows it to tackle any task. \autoref{fig:overview_system} provides a visual representation of the flow within `System`.

![**Flowchart of the `lighter.System`.** A `batch` from the `DataLoader` is processed by `BatchAdapter` to extract `input`, `target` (optional), and `identifier` (optional). The `Model` generates `pred` (predictions) from the `input`. `CriterionAdapter` and `MetricsAdapter` compute loss and metrics, respectively, by applying optional transformations and adapting arguments for the loss and metric functions.  Argument adaptation reorders or names inputs; for example, if a loss function expects `loss_fn(predictions, ground_truth)`, the `CriterionAdapter` maps `pred` to `predictions` and `target` to `ground_truth`. `LoggingAdapter` prepares data for logging. Results, including loss, metrics, and processed data, are returned to the `Trainer`.\label{fig:overview_system}](overview_system.png)


## Lighter's Task-agnosticism and Extensibility

Lighter's flexibility is enabled by two key concepts: adapters and project-specific modules.

### Adapters
Adapters modify how data flows between the dataloader, criterion, metrics, and logging (\autoref{fig:overview_system}). For example, in the configuration below, we specify that the criterion in the training stage should receive `pred` as the first argument and `target` as the second, and that a sigmoid function should be applied to `pred`:

```yaml
adapters:
    train:
        criterion:
            _target_: lighter.adapters.CriterionAdapter
            pred_transforms:   # Apply sigmoid activation to predictions
                _target_: torch.sigmoid
            pred_argument: 0   # Pass 'pred' to criterion's 1st arg
            target_argument: 1 # Pass 'target' to criterion's 2nd arg
```

As a result, Lighter can a wide range of deep learning tasks, from classification and segmentation to self-supervised learning, without requiring changes to the core framework.

### Project-specific modules

Lighter enables users to seamlessly integrate their custom components through project-specific modules. By creating a dedicated project folder structure, users can organize their Python modules containing custom implementations of datasets, models, metrics, transforms, or any other components.

For example, given a project folder `joss_project` with the following structure:

```
joss_project
├── __init__.py
└── models/
    ├── __init__.py
    └── mlp.py
```

This folder will be imported as a module named `project`, which can then be used to reference the components defined within it:


```yaml
project: /path/to/joss_project
system:
    model:
        _target_: project.models.mlp.MLP
        input_size: 784
        num_classes: 10
```

# Research Contributions That Use Lighter

- Foundation model for cancer imaging biomarkers [@Pai2024]
- Vision Foundation Models for Computed Tomography [@Pai2025]

# Acknowledgements

We thank John Zielke for the idea to use adapter design pattern. We thank MONAI team (Wenqi Li, Nic Ma, Yun Liu, Eric Kerfoot) for their continuous support with features and improvements related to MONAI Bundle. 

# References
