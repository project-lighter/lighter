import sys
import importlib
from pathlib import Path

import click
from hydra.utils import instantiate
from loguru import logger

from lightningbringer.configs import Config, init_config


def run(mode, omegaconf_args):
    conf = init_config(omegaconf_args, config_class=Config)
    # Allows the framework to find user-defined, project-specific, classes and their configs
    if conf.project:
        import_project_as_module(conf.project)
    trainer = instantiate(conf.trainer, _convert_="all")
    system = instantiate(conf.system, optimizers=None, _convert_="all")
    # Workaround (including `optimizers=None` above)  TODO: change with Hydra 1.2.0
    # https://github.com/facebookresearch/hydra/issues/1758
    system.optimizers = instantiate(conf.system.optimizers,
                                    system.model.parameters(),
                                    _convert_="all")
    getattr(trainer, mode)(system)


def import_project_as_module(project):
    """Given the path to the project, import it as a module with name 'project'.

    Args:
        project (str): path to the project that will be loaded as module.
    """
    assert isinstance(project, str), "project needs to be a str path"

    # Import project as module with name "project", https://stackoverflow.com/a/41595552
    project_path = Path(project).resolve() / "__init__.py"
    assert project_path.is_file(), f"No `__init__.py` in project `{project_path}`."
    spec = importlib.util.spec_from_file_location("project", str(project_path))
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)
    sys.modules["project"] = project_module
    logger.info(f"Project directory {project} added as a module with name 'project'.")


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
