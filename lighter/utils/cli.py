from typing import Any, Dict

import sys
from functools import partial

import fire
import yaml
from loguru import logger
from monai.bundle.scripts import run
from monai.utils.misc import ensure_tuple

from lighter.utils.dynamic_imports import import_module_from_path

# Trainer methods calls in YAML format with support for command line arguments.
# Waiting for https://github.com/Project-MONAI/MONAI/pull/5854#issuecomment-1384800886.
trainer_methods = {
    "fit": yaml.safe_load(
        """
        fit:
            _method_: >
                $@trainer.fit(model=@fit#model,
                              ckpt_path=@fit#ckpt_path)
            model: "@system"
            ckpt_path: null"""
    ),
    "validate": yaml.safe_load(
        """
        validate:
            _method_: > 
                $@trainer.validate(model=@validate#model,
                                   ckpt_path=@validate#ckpt_path,
                                   verbose=@validate#verbose)
            model: "@system"
            ckpt_path: null
            verbose: True"""
    ),
    "predict": yaml.safe_load(
        """
        predict:
            _method_: >
                $@trainer.predict(model=@predict#model,
                                  ckpt_path=@predict#ckpt_path)
            model: "@system"
            ckpt_path: null"""
    ),
    "test": yaml.safe_load(
        """
        test:
            _method_: >
                $@trainer.test(model=@test#model,
                               ckpt_path=@test#ckpt_path,
                               verbose=@test#verbose)
            model: "@system"
            ckpt_path: null
            verbose: True"""
    ),
    "tune": yaml.safe_load(
        """
        tune:
            _method_: > 
                $@trainer.tune(model=@tune#model,
                               ckpt_path=@tune#ckpt_path,
                               scale_batch_size_kwargs=@tune#scale_batch_size_kwargs,
                               lr_find_kwargs=@tune#lr_find_kwargs,
                               method=@tune#method)
            model: "@system"
            ckpt_path: null
            scale_batch_size_kwargs: null
            lr_find_kwargs: null
            method: fit
    """
    ),
}


def interface():
    fire.Fire({k: partial(run_trainer_method, v) for k, v in trainer_methods.items()})


def run_trainer_method(trainer_method: Dict, **kwargs: Any):
    """Call monai.bundle.run() on a Trainer method. If a project path
    is defined in the config file(s), import it.

    Args:
        trainer_method (Dict): Trainer method to run, collection defined in `trainer_methods`
        **kwargs (Any): keyword arguments for the monai.bundle.run function.
    """
    if "config_file" in kwargs:
        project_imported = False
        # Handle multiple configs
        for config in ensure_tuple(kwargs["config_file"]):
            config = yaml.safe_load(open(config))
            # Import the project as a module
            if "project" in config:
                # Only one config file can specify the project path
                if project_imported:
                    logger.error("`project` must be specified in one config only. Exiting.")
                    sys.exit()
                import_module_from_path("project", config["project"])
                project_imported = True
    # Run the Trainer method
    run(**trainer_method, **kwargs)
