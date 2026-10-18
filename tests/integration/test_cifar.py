"""projects/cifar10 runs every stage: on random images on every pull request, on CIFAR-10 itself in the slow suite.

The CIFAR-10 and CIFAR-100 (projects/lora) examples validate on training images they hold out, not on the test set.
"""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from lighter.engine.runner import Runner
from lighter.utils.dynamic_imports import import_module_from_path

PROJECT = Path(__file__).parents[2] / "projects" / "cifar10"
STAGES = ["fit", "validate", "test", "predict"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A copy of projects/cifar10 as the current folder."""
    folder = tmp_path / "cifar10"
    shutil.copytree(PROJECT, folder, ignore=shutil.ignore_patterns("__pycache__", ".datasets"))
    monkeypatch.chdir(folder)
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    yield folder
    drop()


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_runs_on_random_images(stage, project):
    Runner().run(stage, ["configs/example.yaml", "configs/fake_data.yaml"])
    if stage == "predict":
        assert len(list((project / "predictions").iterdir())) == 64  # FileWriter, one file per image


def test_validation_images_are_held_out_from_the_training_images(project, monkeypatch):
    import_module_from_path("project", project)
    dataset = importlib.import_module("project.dataset")
    monkeypatch.setattr(dataset, "CIFAR10", lambda root, train, **kwargs: list(range(1000 if train else 200)))
    train, val, test = (list(dataset.cifar10_split("unused", split)) for split in ("train", "val", "test"))
    assert len(val) == 100 and not set(train) & set(val) and sorted(train + val) == list(range(1000))
    assert test == list(range(200))
    assert list(dataset.cifar10_split("unused", "val")) == val  # the same split every time


def test_lora_validation_images_are_held_out_from_the_training_images(project, monkeypatch):
    import_module_from_path("project", PROJECT.parent / "lora")  # the fixture drops it again afterwards
    dataset = importlib.import_module("project.dataset")
    monkeypatch.setattr(dataset.datasets, "CIFAR100", lambda root, train, download: list(range(1000 if train else 200)))
    train, val, test = (dataset.CIFAR100Dataset(split=split) for split in ("train", "val", "test"))
    assert len(val) == 100 and not set(train.indices) & set(val.indices)
    assert sorted(train.indices + val.indices) == list(range(1000)) and len(test) == 200


@pytest.mark.slow
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_runs_on_cifar10(stage, monkeypatch):
    """Downloads CIFAR-10 into projects/cifar10/.datasets once."""
    monkeypatch.chdir(PROJECT)
    Runner().run(stage, ["configs/example.yaml", "../../tests/integration/test_overrides.yaml"])
