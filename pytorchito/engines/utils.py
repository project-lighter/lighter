from pytorchito.engines.trainer import Trainer
from pytorchito.engines.validator_tester import Tester
from pytorchito.engines.inferer import Inferer


ENGINES = {
    'train': Trainer,
    'test': Tester,
    'infer': Inferer
}



def init_engine(mode, omegaconf_args):
    pass