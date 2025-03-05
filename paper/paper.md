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
    dropping-particle: JWL
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
 - name: Department of Cardiovascular Radiology and Nuclear Medicine, Technical University of Munich, School of Medicine and Health, German Heart Center, TUM University Hospital, Germany
   index: 6
date: 24 February 2025
bibliography: paper.bib

---

# Summary

Lighter is an open-source Python [framework](https://github.com/project-lighter/lighter) for deep learning research that builds upon PyTorch Lightning [@Falcon_PyTorch_Lightning_2019] and the [MONAI Bundle configuration](https://docs.monai.io/en/stable/config_syntax.html#) [@Cardoso_MONAI_An_open-source_2022]. With its declarative YAML-based configuration system that is both transparent and self-documenting, Lighter aims to streamline deep learning research. Researchers define experimental protocols—including neural network architectures, optimization strategies, data pipelines, and evaluation metrics—through structured configuration files, which effectively decouple scientific hypotheses from implementation details. This separation reduces boilerplate code while preserving complete experimental control. Lighter enables reproducibility through comprehensive configuration snapshots that document all experimental parameters and dependencies. A modular adapter system and support for project-specific extensions ensure the extensibility of the framework enabling researchers to implement specialized methodologies without modifying core framework components. By abstracting the engineering complexities of deep learning experimentation, Lighter allows researchers to focus on scientific innovation, accelerate hypothesis testing, and facilitate rigorous validation of research findings across application domains.

# Statement of Need

Lighter is designed to address several key challenges in deep learning experimentation:

1.  **Boilerplate Code:** Writing code for training loops, data loading, metric calculations, and experiment setups is repetitive and can vary greatly between projects. *Lighter abstracts these repetitive tasks, exposing only the components that differ across projects.*

2.  **Experiment Management:** Handling numerous hyperparameters and configurations across various experiments can become cumbersome and error-prone. *Lighter offers organized configuration through YAML files, providing a **centralized record of all experiment parameters**.*

3.  **Reproducibility:** Reproducing experiments from different implementations can be challenging. *Lighter's **self-contained configuration files** serve as comprehensive documentation, facilitating the exact recreation of experimental setups.*

4.  **Collaboration:** Collaborating on experiments often requires understanding complex codebases. *Lighter enhances collaboration by using standardized configurations, making it easier to share and reuse experiment setups within and across research teams.*

5.  **Slowed Iteration:** The cumulative effect of these challenges slows down the research iteration cycle. *Lighter accelerates iteration by streamlining the experiment setup process, allowing researchers to focus on core experiment choices without being bogged down by infrastructure concerns.*


# Design

Lighter's architecture is built upon three fundamental components that work together to streamline deep learning experimentation:

1.  **`Config`**: This component serves as the experiment's blueprint, parsing and validating YAML configuration files that comprehensively define all aspects of the experimental setup. Within these configuration files, researchers specify the `System` and `Trainer` parameters, creating a self-documenting record of the experiment.

2.  **`System`**: The `System` orchestrates the main building blocks of an experiment: model, optimizer, scheduler, loss function, evaluation metrics, and dataloaders. It implements the logic controlling how these components interact during training, validation, testing, and inference phases, that is modifiable through [adapters](#adapters).

3. **`Trainer`**: PyTorch Lightning's robust `Trainer` class handles the technical aspects of the training process. It manages advanced features such as distributed training across multiple GPUs, mixed precision computation for memory efficiency, and checkpoint management for experiment continuity. Lighter employs the `Trainer` to execute the experimental protocol defined by the `System`.

![**Lighter Overview.** Lighter revolves around three main components -- `Trainer,` `System` and `Config`, which contains the definition for the former two. `Config` leverages MONAI's `ConfigParser` for parsing the user-defined YAML configuration files, and its features are used by Runner to instantiate the `System` and `Trainer`. `Trainer` is used directly from PyTorch Lightning, whereas `System` inherits from `LightningModule`, ensuring its compatibility with `Trainer` while implementing a logic generalizable to any task or type of data. Finally, `Runner` runs the paired `Trainer` and `System` for a particular stage (e.g., fit or test).](overview_all.png)

## System

The `System` component in Lighter serves as a comprehensive abstraction layer that encapsulates the essential experimental elements—including neural network architectures, optimization strategies, learning rate schedulers, loss functions, evaluation metrics, and data pipelines. This component implements a generalized data flow paradigm that orchestrates interactions between these elements during the experimental lifecycle. A distinguishing feature of the `System` is its configurable nature through the [adapter](#adapters) mechanism, which provides the flexibility required to address diverse research tasks without architectural modifications. The systematic flow of information through the `System` is illustrated in \autoref{fig:overview_system}, demonstrating how data traverses from input through model inference to evaluation and logging.

![**Flowchart of the `lighter.System`.** A `batch` from the `DataLoader` is processed by `BatchAdapter` to extract `input`, `target` (optional), and `identifier` (optional). The `Model` generates `pred` (predictions) from the `input`. `CriterionAdapter` and `MetricsAdapter` compute loss and metrics, respectively, by applying optional transformations and adapting arguments for the loss and metric functions.  Argument adaptation reorders or names inputs; for example, if a loss function expects `loss_fn(predictions, ground_truth)`, the `CriterionAdapter` maps `pred` to `predictions` and `target` to `ground_truth`. `LoggingAdapter` prepares data for logging. Results, including loss, metrics, and processed data, are returned to the `Trainer`.\label{fig:overview_system}](overview_system.png)


## Adaptability Through Modular Design

Lighter achieves task-agnostic flexibility through two key concepts: adapters and project-specific module integration.

### Adapters
The adapter pattern implements a transformation layer between core system components, enabling customized data flow between the dataloader, criterion, metrics computation, and logging subsystems (\autoref{fig:overview_system}). This abstraction allows researchers to modify component interactions without altering the underlying framework. Consider the following configuration excerpt that demonstrates the adapter's capability to transform prediction outputs and remap function arguments:

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

As a result, Lighter can execute a wide range of deep learning tasks, from classification and segmentation to self-supervised learning, without requiring changes to the core framework.

### Project-specific modules

Lighter provides the integration of project-specific implementations through a modular project structure. Researchers can use their custom components—including novel architectures, specialized datasets, task-specific metrics, and domain-adapted transforms—within a structured project directory. This organization promotes code reusability and maintains a clear separation between framework functionality and project-specific implementations.

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

Lighter has enabled advancements in medical imaging research:

- Foundation model for cancer imaging biomarkers [@Pai2024]
- Vision Foundation Models for Computed Tomography [@Pai2025]

# Acknowledgments

We thank John Zielke for the idea to use adapter design pattern. We thank the MONAI team (Wenqi Li, Nic Ma, Yun Liu, Eric Kerfoot) for their continuous support with features and improvements related to MONAI Bundle. 

# References
