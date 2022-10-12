import sys

from hydra.utils import instantiate
from loguru import logger

from lightningbringer.config import init_config_from_cli


def run(mode, args):
    conf, method_args = init_config_from_cli(args, log=True)
    trainer = init_trainer(conf)
    system = init_system(conf, mode)
    # Run the mode (train, validate, test, etc.)
    getattr(trainer, mode)(model=system, **method_args)


def init_trainer(conf):
    return instantiate(conf.trainer, _convert_="all")


def init_system(conf, mode=None):
    # Don't instantiate datasets and samplers that won't be used in the run
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
    from omegaconf import DictConfig
    assert isinstance(conf.system.optimizers, DictConfig), "One optimizer!"
    assert isinstance(conf.system.schedulers, (DictConfig, type(None))), "One scheduler!"
    system.optimizers = instantiate(conf.system.optimizers,
                                    params=system.model.parameters(),
                                    _convert_="all")

    # Handles the PL scheduler dicts. This won't be needed once the Hydra issue is resolved.
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
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
