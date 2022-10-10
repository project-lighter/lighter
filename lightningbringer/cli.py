import click

from lightningbringer.runner import run


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


# Predict
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
