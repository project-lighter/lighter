import click
from pytorchito.engines.utils import init_engine


# Interface
@click.group()
def interface():
    """pytorchito"""


# Train
@interface.command(help="Train a model.")
@click.argument("omegaconf_args", nargs=-1)
def train(omegaconf_args):
    init_engine('train', omegaconf_args).run()


# Test
@interface.command(help="Test a trained model.")
@click.argument("omegaconf_args", nargs=-1)
def test(omegaconf_args):
    init_engine('test', omegaconf_args).run()


# Infer
@interface.command(help="Infer with a trained model.")
@click.argument("omegaconf_args", nargs=-1)
def infer(omegaconf_args):
    init_engine('infer', omegaconf_args).run()


if __name__ == "__main__":
    interface()
