"""Tests for running CIFAR training to verify integrity of the pipeline"""

from pathlib import Path

import pytest

from lighter.engine.runner import Runner, Stage


@pytest.mark.parametrize(
    ("stage", "config"),
    [
        (
            Stage.FIT,
            "configs/example.yaml",
        ),
        (
            Stage.TEST,
            "configs/example.yaml",
        ),
        (
            Stage.PREDICT,
            "configs/example.yaml",
        ),
    ],
)
@pytest.mark.slow
def test_trainer_stage(stage: Stage, config: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the specified stage using the given configuration.
    Args:
        stage: The stage to run (e.g., Stage.FIT, Stage.TEST, Stage.PREDICT).
        config: Path to the configuration file.
        monkeypatch: Pytest fixture for changing working directory.
    """
    # Change to CIFAR10 project directory for auto-discovery
    project_dir = Path(__file__).parent.parent.parent / "projects" / "cifar10"
    monkeypatch.chdir(project_dir)

    # Paths relative to project directory
    test_overrides = "../../tests/integration/test_overrides.yaml"

    runner = Runner()
    runner.run(stage, [config, test_overrides])
    # Runner no longer stores trainer, just verify it completed without error
