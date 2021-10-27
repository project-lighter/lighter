import click
from hydra.utils import instantiate

from lightningbringer.config import init_config


def run(mode, omegaconf_args):
    conf = init_config(omegaconf_args, log=True)
    print(type(conf))
    # Don't instantiate datasets that won't be used in the run
    train_dataset, val_dataset, test_dataset = None, None, None
    if mode == "fit":
        train_dataset = conf.system.train_dataset
        val_dataset = conf.system.val_dataset
    elif mode == "validate":
        val_dataset = conf.system.val_dataset
    elif mode == "test":
        test_dataset = conf.system.test_dataset
    else:
        raise NotImplementedError(f"No behavior for {mode}.")

    # Instantiate the Trainer and the System
    trainer = instantiate(conf.trainer, _convert_="all")
    system = instantiate(conf.system,
                         optimizers=None,
                         train_dataset=train_dataset,
                         val_dataset=val_dataset,
                         test_dataset=test_dataset,
                         _convert_="all")

    # Workaround (including `optimizers=None` above)  TODO: change with Hydra 1.2.0
    # https://github.com/facebookresearch/hydra/issues/1758
    system.optimizers = instantiate(conf.system.optimizers,
                                    system.model.parameters(),
                                    _convert_="all")
    getattr(trainer, mode)(system)


################### Command Line Interface ###################


# Interface
@click.group()
def interface():
    """lightningbringer"""


# Train
@interface.command(help="TODO")
@click.argument("omegaconf_args", nargs=-1)
def train(omegaconf_args):
    run('fit', omegaconf_args)


# Validate
@interface.command(help="TODO")
@click.argument("omegaconf_args", nargs=-1)
def validate(omegaconf_args):
    run('validate', omegaconf_args)


# Test
@interface.command(help="TODO")
@click.argument("omegaconf_args", nargs=-1)
def predict(omegaconf_args):
    run('predict', omegaconf_args)


# Test
@interface.command(help="TODO")
@click.argument("omegaconf_args", nargs=-1)
def test(omegaconf_args):
    run('test', omegaconf_args)


# Tune
@interface.command(help="TODO")
@click.argument("omegaconf_args", nargs=-1)
def tune(omegaconf_args):
    run('tune', omegaconf_args)


if __name__ == "__main__":
    interface()
