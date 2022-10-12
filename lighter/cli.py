import click

from lighter.runner import run


# Interface
@click.group()
def interface():
    """lighter"""


# Train
@interface.command(help="TODO")
@click.argument("args", nargs=-1)
def train(args):
    run('fit', args)


# Validate
@interface.command(help="TODO")
@click.argument("args", nargs=-1)
def validate(args):
    run('validate', args)


# Predict
@interface.command(help="TODO")
@click.argument("args", nargs=-1)
def predict(args):
    run('predict', args)


# Test
@interface.command(help="TODO")
@click.argument("args", nargs=-1)
def test(args):
    run('test', args)


# Tune
@interface.command(help="TODO")
@click.argument("args", nargs=-1)
def tune(args):
    run('tune', args)


if __name__ == "__main__":
    interface()
