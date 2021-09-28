from pytorchito.utils import communication
from pytorchito.utils.importing import import_attr
from pytorchito.configs.utils import init_config
from pytorchito.configs.config import Config


def init_engine(mode, omegaconf_args):
    """Instantiate the train/val/test/infer engine.

    Args:
        mode (str): train/val/test/infer mode of running.
        omegaconf_args (list): list of arguments passed in pytorchito's CLI

    Returns:
        object: instance of the engine specified under the `engine` option in config.
    """
    communication.init_distributed()
    conf = init_config(omegaconf_args, config_class=Config)
    engine = import_attr(conf[mode].engine)
    return engine(conf)
