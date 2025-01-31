"""Tests for running CIFAR training to verify integrity of the pipeline"""
import pytest

from lighter.engine.runner import Runner, Stage

test_overrides = "./tests/integration/test_overrides.yaml"


@pytest.mark.parametrize(
    ("stage", "config"),
    [
        (
            Stage.FIT,
            "./projects/cifar10/experiments/example.yaml",
        ),
        (
            Stage.TEST,
            "./projects/cifar10/experiments/example.yaml",
        ),
        (
            Stage.PREDICT,
            "./projects/cifar10/experiments/example.yaml",
        ),
    ],
)
@pytest.mark.slow
def test_trainer_stage(stage: Stage, config: str):
    """
    Test the specified stage using the given configuration.
    Args:
        stage: The stage to run (e.g., "fit", "test", "predict").
        config: Path to the configuration file.
    """
    runner = Runner()
    runner.run(stage, config=f"{config},{test_overrides}")
    assert runner.trainer.state.finished, f"Stage {stage} did not finish successfully."
