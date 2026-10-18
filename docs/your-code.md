---
title: Your code and project.
---

# Your code and `project.`

Put a file named `__lighter__.py` in the folder that holds your code. Lighter imports that folder as a package
named `project`, so a config names your classes `project.<file>.<Class>`:

```text
quickstart/
├── __lighter__.py      # marks the folder; Lighter never runs it, so it can be empty
├── model.py            # class Regressor  ->  _target_: project.model.Regressor
├── data.py             # class RandomData ->  _target_: project.data.RandomData
└── config.yaml
```

Subfolders work the same way: `project/nets/unet.py` is `project.nets.unet`. `__init__.py` files are optional;
add one only for code that should run on import, or to shorten names (`from .unet import UNet` in
`nets/__init__.py` makes `project.nets.UNet` work).

## Where Lighter looks

Lighter looks for `__lighter__.py` in each config file's folder and its parents, then in the current folder and
its parents. So these all find the same project:

```bash
lighter fit config.yaml                            # from the project folder
lighter fit configs/train.yaml                     # a config in a subfolder
lighter fit ../my_project/configs/train.yaml       # from anywhere else
```

Config files from two different projects in one run are an error. Without any `__lighter__.py`, `_target_` can
still name installed packages (`torch.nn.Linear`, or your own package after `pip install -e .`).

## Imports inside your code

Your files are modules of the package `project`, so they import each other with relative imports
(`from .data import RandomData`) or by full name (`from project.data import RandomData`). A bare
`import data` fails: Lighter does not put the current folder on `sys.path`, so a file of yours can never hide
an installed package with the same name, and a run does not depend on the folder you start it from.

DataLoader workers that start with `spawn` (the default on macOS and Windows) can use your classes:
Lighter sends the `project` code to them along with the dataset.

## When something is off

A `_target_` without `project.`, a missing marker or a misspelled class name each print the config line and
what to write instead. [Troubleshooting](troubleshooting.md) shows the messages.
