import click

from lighter.runner import run


# Interface
@click.group()
def interface():
    """Lighter - config-based wrapper for PyTorch Lightning for
    deep learning with little to no coding required.
    """


# Train
@interface.command()
@click.argument("args", nargs=-1)
def train(args):
    run('fit', args)


# Validate
@interface.command()
@click.argument("args", nargs=-1)
def validate(args):
    run('validate', args)


# Predict
@interface.command()
@click.argument("args", nargs=-1)
def predict(args):
    run('predict', args)


# Test
@interface.command()
@click.argument("args", nargs=-1)
def test(args):
    run('test', args)


# Tune
@interface.command()
@click.argument("args", nargs=-1)
def tune(args):
    run('tune', args)


if __name__ == "__main__":
    interface()
