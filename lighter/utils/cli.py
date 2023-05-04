import sys
from functools import partial

import fire

from lighter.utils.runner import run_trainer_method


def interface():
    """Defines the command line interface for running Trainer's methods. The available methods are:
    - fit
    - validate
    - test
    - predict
    - tune
    """
    commands = {}
    # All Trainer methods' names
    for name in ["fit", "validate", "test", "predict", "tune"]:
        # Creates command for fire to run. The command is the Trainer method's name. Fire passes
        # **kwargs to the function.
        commands[name] = partial(run_trainer_method, name)

    fire.Fire(commands)
