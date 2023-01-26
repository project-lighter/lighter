"""Tests for running CIFAR training to verify integrity of the pipeline"""
import pytest

from lighter.utils.cli import run_trainer_method, trainer_methods

test_overrides = "./tests/integration/test_overrides.yaml"


@pytest.mark.parametrize(
    ("mode", "config_file"),
    [("fit", "./projects/cifar10/experiments/monai_bundle_prototype.yaml")],
)
@pytest.mark.slow
def test_trainer_method(mode: str, config_file: str):
    """Test trainer method for different mode configurations"""
    kwargs = {"config_file": config_file, "args_file": test_overrides}

    func_return = run_trainer_method(trainer_methods[mode], **kwargs)
    assert func_return == None
