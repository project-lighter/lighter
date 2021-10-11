from hydra.utils import instantiate

from pytorchito.configs import Config, init_config


def run(mode, omegaconf_args):
    """TODO

    Args:
        mode (str): train/val/test/infer mode of running.
        omegaconf_args (list): list of arguments passed in pytorchito's CLI

    Returns:
        object: instance of the engine specified under the `engine` option in config.
    """
    conf = init_config(omegaconf_args, config_class=Config)

    trainer = instantiate(conf.trainer)
    system = instantiate(conf.system, optimizers=None)
    # Workaround for https://tinyurl.com/psv9mbub (including the `optimizers=None` above)
    system.optimizers = instantiate(conf.system.optimizers, system.model.parameters())

    assert mode in ["fit", "validate", "test", "tune"], f"Trainer does not have a '{mode}' method."
    getattr(trainer, mode)(system)
