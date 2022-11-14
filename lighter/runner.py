import sys
from typing import Any

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from lighter.config import init_config_from_cli


def run(trainer_method_name: str, args: list) -> None:
    """Initializes the trainer and the system, passes the system to the trainer
    and calls the specified trainer's method.

    Args:
        trainer_method_name (str): trainer's method that will be used. It's name is also
            used to let the system know in which mode should it run.
        args (list): command-line arguments passed when running Lighter's CLI.
    """
    conf, method_args = init_config_from_cli(args, log=True)
    trainer = init_trainer(conf)
    system = init_system(conf, mode=trainer_method_name)
    # Run the mode (train, validate, test, etc.)
    getattr(trainer, trainer_method_name)(model=system, **method_args)


def init_trainer(conf: DictConfig) -> Any:
    """Initialize the trainer given it's definition in the config.

    Args:
        conf (DictConfig): the experiment's config.

    Returns:
        Any: the trainer instance of a particular class.
    """
    return instantiate(conf.trainer, _convert_="all")


def init_system(conf: DictConfig, mode: str = None) -> Any:
    """Initialize the trainer given its definition in the config.

    Args:
        conf (DictConfig): the experiment's config.
        mode (str, optional): Mode in which the system will be initialized. Defaults to None.

    Returns:
        Any: the sytem instance of a particular class.
    """
    # Datasets and samplers not used in the run won't be instantiated
    if mode in ["fit", "tune"]:
        conf.system.test_dataset = None
        conf.system.test_sampler = None
    elif mode == "validate":
        conf.system.train_dataset = conf.system.test_dataset = None
        conf.system.train_sampler = conf.system.test_sampler = None
    elif mode == "test":
        conf.system.train_dataset = conf.system.val_dataset = None
        conf.system.train_sampler = conf.system.val_sampler = None
    elif mode is not None:
        logger.error(f"'{mode}' mode does not exist. Exiting.")
        sys.exit()

    # Instantiate the System
    system = instantiate(conf.system, optimizers=None, schedulers=None, _convert_="all")

    ################ https://github.com/facebookresearch/hydra/issues/1758 ################
    # This issue prevents us from referencing other objects through config. For example,
    # the optimizer requires model's parameters, and instead of referring to the model,
    # we instantiate the optimizer separately once the model has been instantiated.
    # The same goes for the schedulers. Currently, because of this behavior, only
    # one optimizer and scheduler are allowed. TODO: change it when Hydra fixes this issue.
    # This workaround includes the `optimizers=None` and `schedulers=None` above).
    assert isinstance(conf.system.optimizers, DictConfig), "One optimizer!"
    assert isinstance(conf.system.schedulers, (DictConfig, type(None))), "One scheduler!"
    system.optimizers = instantiate(conf.system.optimizers,
                                    params=system.model.parameters(),
                                    _convert_="all")

    # Handles the PL scheduler dicts. This won't be needed once the Hydra issue is resolved.
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
    if conf.system.schedulers is not None:
        if "scheduler" in conf.system.schedulers:
            system.schedulers = instantiate(conf.system.schedulers,
                                            scheduler={"optimizer": system.optimizers},
                                            _convert_="all")
        else:
            system.schedulers = instantiate(conf.system.schedulers,
                                            optimizer=system.optimizers,
                                            _convert_="all")

    #######################################################################################
    return system
