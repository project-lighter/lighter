from pytorchito.utils.io import import_attr
from pytorchito.configs.utils import init_config
from pytorchito.configs.config import Config


def init_engine(mode, omegaconf_args):
    conf = init_config(omegaconf_args, config_class=Config)
    engine = import_attr(conf[mode].engine)
    return engine(conf)
