import click
from pytorchito.runner import run


# Interface
@click.group()
def interface():
    """pytorchito"""


# Train
@interface.command(help="Train a model.")
@click.argument("omegaconf_args", nargs=-1)
def train(omegaconf_args):
    run('fit', omegaconf_args)


# Validate
@interface.command(help="TODO")
@click.argument("omegaconf_args", nargs=-1)
def validate(omegaconf_args):
    run('validate', omegaconf_args)

# Test
@interface.command(help="Test a trained model.")
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
