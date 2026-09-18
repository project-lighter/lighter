"""Execution inputs have observable effects before scientific construction."""

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from lighter import Runner


class SeedTask(LightningModule):
    def __init__(self):
        super().__init__()
        self.draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    def predict_step(self, batch, batch_idx):
        return self.draw


def loader():
    return DataLoader([1], batch_size=None)


def fail_if_constructed():
    raise AssertionError("An overridden or inactive stage argument was constructed")


def recipe(tmp_path):
    return {
        "model": {"_target_": f"{__name__}.SeedTask"},
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "default_root_dir": str(tmp_path),
        },
        "args": {"predict": {"dataloaders": {"_target_": f"{__name__}.loader"}}},
    }


@pytest.mark.parametrize("seed", [0, 11, 42])
def test_seed_precedes_project_import_and_model_construction(tmp_path, monkeypatch, seed):
    monkeypatch.setenv("PL_GLOBAL_SEED", "999")
    config = recipe(tmp_path)
    config["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    imported = [random.random(), float(np.random.random()), float(torch.rand(()))]
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    observed = []

    def project_import():
        observed.extend([random.random(), float(np.random.random()), float(torch.rand(()))])

    with patch("lighter.engine.runner.ProjectImporter.auto_discover_and_import", side_effect=project_import):
        result = Runner().run("predict", [config], dataloaders=loader())
    assert observed == imported
    assert result == [expected]
    assert str(seed) == __import__("os").environ["PL_GLOBAL_SEED"]
    assert __import__("os").environ["PL_SEED_WORKERS"] == "1"


def test_default_seed_is_stable_and_not_previous_run_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "321")
    first = Runner().run("predict", [recipe(tmp_path)], dataloaders=loader())
    monkeypatch.setenv("PL_GLOBAL_SEED", "654")
    second = Runner().run("predict", [recipe(tmp_path)], dataloaders=loader())
    assert first == second


@pytest.mark.parametrize("seed", [True, None, -1, 2**32, 1.5, "42", "$42"])
def test_invalid_seed_fails_before_import_or_construction(tmp_path, seed):
    config = recipe(tmp_path)
    config["seed"] = seed
    with patch("lighter.engine.runner.ProjectImporter.auto_discover_and_import") as importer:
        with pytest.raises(ValueError, match="seed"):
            Runner().run("predict", [config], dataloaders=loader())
    importer.assert_not_called()


def test_configured_stage_arguments_supply_native_direct_loader(tmp_path):
    config = recipe(tmp_path)
    config["args"]["test"] = {"dataloaders": {"_target_": f"{__name__}.fail_if_constructed"}}
    result = Runner().run("predict", [config])
    assert len(result) == 1


def test_explicit_stage_argument_overrides_recipe_without_constructing_it(tmp_path):
    config = recipe(tmp_path)
    config["args"]["predict"]["dataloaders"] = {"_target_": f"{__name__}.fail_if_constructed"}
    config["args"]["predict"]["return_predictions"] = True
    assert Runner().run("predict", [config], dataloaders=loader(), return_predictions=False) is None


@pytest.mark.parametrize("arguments", [[], {"predict": []}, {"predict": {"model": "replacement"}}])
def test_invalid_stage_arguments_fail_clearly(tmp_path, arguments):
    config = recipe(tmp_path)
    config["args"] = arguments
    with pytest.raises((ValueError, TypeError), match="args|model"):
        Runner().run("predict", [config], dataloaders=loader())


def test_override_is_authoritative_inside_references_and_import_branches(tmp_path):
    config = recipe(tmp_path)
    config["args"]["predict"] = {
        "dataloaders": {
            "_target_": f"{__name__}.fail_if_constructed",
            "side_effect": "$import no_such_overridden_module",
        },
        "return_predictions": "$@args::predict::dataloaders is not None",
    }
    config["args"]["test"] = {"inactive_import": "$import no_such_inactive_module"}
    assert len(Runner().run("predict", [config], dataloaders=loader())) == 1


def test_stage_argument_error_names_installed_lightning_signature(tmp_path):
    config = recipe(tmp_path)
    with pytest.raises(TypeError, match="Trainer.predict.*installed Lightning.*unknown_native_argument"):
        Runner().run("predict", [config], unknown_native_argument=True)
