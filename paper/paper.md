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

Lighter is an open-source Python deep learning [framework](https://github.com/project-lighter/lighter) that builds upon PyTorch Lightning [@Falcon_PyTorch_Lightning_2019] and [MONAI Bundle configuration](https://docs.monai.io/en/stable/config_syntax.html#) [@Cardoso_MONAI_An_open-source_2022]. It streamlines deep learning research through YAML-based configuration that decouples experiment setup from implementation details. Researchers define models, datasets, and other components via structured configuration files, reducing boilerplate while maintaining control. The framework enhances reproducibility through configuration snapshots and supports extensibility via adapters and project-specific modules. By abstracting engineering complexities, Lighter allows researchers to focus on innovation, accelerate hypothesis testing, and facilitate rigorous validation across domains.

# Statement of Need

Lighter is designed to address several key challenges in deep learning experimentation:

1.  **Boilerplate Code:** Writing code for training loops, data loading, metric calculations, and experiment setups is repetitive and can vary greatly between projects. *Lighter abstracts these repetitive tasks, exposing only the components that differ across projects.*

2.  **Experiment Management:** Handling numerous hyperparameters and configurations across various experiments can become cumbersome and error-prone. *Lighter offers organized configuration through YAML files, providing a **centralized record of all experiment parameters**.*

3.  **Reproducibility:** Reproducing experiments from different implementations can be challenging. *Lighter's **self-contained configuration files** serve as comprehensive documentation, facilitating the exact recreation of experimental setups.*

4.  **Collaboration:** Collaborating on experiments often requires understanding complex codebases. *Lighter enhances collaboration by using standardized configurations, making it easier to share and reuse experiment setups within and across research teams.*

5.  **Slowed Iteration:** The cumulative effect of these challenges slows down the research iteration cycle. *Lighter accelerates iteration by streamlining the experiment setup process, allowing researchers to focus on core experiment choices without being bogged down by infrastructure concerns.*


# Design

Lighter is built upon three fundamental components (\autoref{fig:overview_all}):

1.  **`Config`**: serves as the experiment's blueprint, parsing and validating YAML configuration files that define all aspects of the experimental setup. Within these configuration files, researchers specify the `System` and `Trainer` parameters, creating a self-documenting record of the experiment.

2.  **`System`**: encapsulates the model, optimizer, scheduler, loss function, metrics, and dataloaders. Importantly, it implements the flow between them that can be customized through [adapters](#adapters) (\autoref{fig:overview_system}).

3. **`Trainer`**:  PyTorch Lightning's `Trainer` handles aspects like distributed or mixed-precision training and checkpoint management. Lighter uses it to execute the protocol defined by the `System`.

![**Lighter Overview.** `Config` leverages MONAI's `ConfigParser` for parsing the user-defined YAML configuration files, and its features are used by Runner to instantiate the `System` and `Trainer`. `Trainer` is used directly from PyTorch Lightning, whereas `System` inherits from `LightningModule`, ensuring its compatibility with `Trainer` while implementing a logic generalizable to any task or type of data. Finally, `Runner` runs the paired `Trainer` and `System` for a particular stage (e.g., fit or test).\label{fig:overview_all}](overview_all.png)

![**Flowchart of the `lighter.System`.** A `batch` from the `DataLoader` is processed by `BatchAdapter` to extract `input`, `target` (optional), and `identifier` (optional). The `Model` generates `pred` (predictions) from the `input`. `CriterionAdapter` and `MetricsAdapter` compute loss and metrics, respectively, by applying optional transformations and routing arguments for the loss and metric functions. Results, including loss, metrics, and other data prepared for logging by the `LoggingAdapter` are returned to the `Trainer`.\label{fig:overview_system}](overview_system.png)


## Adaptability Through Modular Design

### Adapters

The adapter pattern creates an interface between core system components, allowing customization of the data flow. By configuring adapters, users can modify how components interact without changing the underlying code. Consequently, Lighter is task-agnostic and applicable to tasks ranging from classification to self-supervised learning. For example, you can implement the following criterion adapter to apply sigmoid activation to predictions and route the data to a criterion's respective arguments:

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


### Project-specific modules

Lighter's modular design lets researchers add custom components in organized project directories. For example, a project folder like:

```
joss_project
├── __init__.py
└── models/
    ├── __init__.py
    └── mlp.py
```

is imported as a module named `project`, with its components accessible in configuration:

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

# Acknowledgments

We thank John Zielke for the adapter design pattern idea. We thank Wenqi Li, Nic Ma, Yun Liu, and Eric Kerfoot for their continuous support with MONAI Bundle. 

# References
