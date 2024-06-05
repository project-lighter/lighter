"""Tests for running CIFAR training to verify integrity of the pipeline"""

import pytest

from lighter.utils.cli import run_trainer_method

test_overrides = "./tests/integration/test_overrides.yaml"


@pytest.mark.parametrize(
    ("method_name", "config_file"),
    [
        (  # Method name
            "fit",
            # Config fiile
            "./projects/cifar10/experiments/monai_bundle_prototype.yaml",
        )
    ],
)
@pytest.mark.slow
def test_trainer_method(method_name: str, config_file: str):
    """ """
    kwargs = {"config_file": config_file, "args_file": test_overrides}

    func_return = run_trainer_method(method_name, **kwargs)
    assert func_return is None
