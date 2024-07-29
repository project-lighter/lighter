"""Tests for running CIFAR training to verify integrity of the pipeline"""

import pytest

from lighter.utils.runner import run

test_overrides = "./tests/integration/test_overrides.yaml"


@pytest.mark.parametrize(
    ("method_name", "config"),
    [
        (  # Method name
            "fit",
            # Config fiile
            "./projects/cifar10/experiments/monai_bundle_prototype.yaml",
        ),
        (  # Method name
            "test",
            # Config fiile
            "./projects/cifar10/experiments/monai_bundle_prototype.yaml",
        ),
        (  # Method name
            "predict",
            # Config fiile
            "./projects/cifar10/experiments/monai_bundle_prototype.yaml",
        ),
    ],
)
@pytest.mark.slow
def test_trainer_method(method_name: str, config: str):
    """ """

    kwargs = {"config": [config, test_overrides]}
    func_return = run(method_name, **kwargs)
    assert func_return is None
