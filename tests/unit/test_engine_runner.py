"""Unit tests for building and running stages in lighter/engine/runner.py"""

import os
import re
import subprocess
import sys
import warnings

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from sparkwheel import TargetNotFoundError
from torch.utils.data import DataLoader, TensorDataset

import lighter
from lighter import LighterDataModule, Runner, build
from lighter.engine.runner import find_project_root, load_config

TRAINER = {
    "_target_": "pytorch_lightning.Trainer",
    "accelerator": "cpu",
    "max_epochs": 1,
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


class ResultTask(LightningModule):
    """A tiny module whose stage results are easy to check."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight * batch[0].float().mean()

    def validation_step(self, batch, batch_idx):
        self.log("score", batch[0].float().mean(), batch_size=len(batch[0]))

    test_step = validation_step

    def predict_step(self, batch, batch_idx):
        return batch[0] * 2

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def train_dataloader(self):
        return result_loader()

    val_dataloader = test_dataloader = predict_dataloader = train_dataloader


def result_loader():
    return DataLoader(TensorDataset(torch.arange(5)), batch_size=2)


def result_data():
    return LighterDataModule(**{f"{stage}_dataloader": result_loader() for stage in ("train", "val", "test", "predict")})


def config(**extra):
    return {"model": {"_target_": f"{__name__}.ResultTask"}, "trainer": dict(TRAINER), **extra}


def test_load_config_applies_inputs_in_order():
    loaded = load_config(config(), "trainer::max_epochs=5", {"trainer": {"devices": 1}})
    assert loaded.get("trainer::max_epochs") == 5
    assert loaded.get("trainer::devices") == 1


def test_build_returns_objects_without_running():
    model, trainer, datamodule = build(config())
    assert isinstance(model, ResultTask)
    assert isinstance(trainer, Trainer)
    assert datamodule is None
    assert trainer.global_step == 0 and model.weight.item() == 1.0


def test_build_resolves_the_data_key():
    _, _, datamodule = build(config(data={"_target_": f"{__name__}.result_data"}))
    assert isinstance(datamodule, LighterDataModule)


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("external_data", [False, True])
def test_run_returns_the_native_stage_result(tmp_path, stage, external_data):
    options = {k: v for k, v in TRAINER.items() if k != "_target_"}
    native_args = {"datamodule": result_data()} if external_data else {}
    extra = {"data": {"_target_": f"{__name__}.result_data"}} if external_data else {}
    expected = getattr(Trainer(**options), stage)(ResultTask(), **native_args)
    actual = Runner().run(stage, [config(**extra)])
    if stage == "predict":
        assert torch.equal(torch.cat(actual), torch.arange(5) * 2)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    elif stage == "fit":
        assert actual is expected is None
    else:
        assert actual == expected == [{"score": 2.0}]


def test_run_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage 'train'"):
        Runner().run("train", [config()])


def test_model_must_be_a_lightning_module():
    with pytest.raises(TypeError, match="`model` must build a LightningModule, got Linear"):
        build(config(model={"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1}))


def test_trainer_without_target_explains_the_fix():
    with pytest.raises(TypeError, match="Add a `_target_` to `trainer:`"):
        build({"model": {"_target_": f"{__name__}.ResultTask"}, "trainer": {"max_epochs": 1}})


def test_lightning_package_modules_are_accepted():
    import lightning.pytorch as L

    class LightningTask(L.LightningModule):
        pass

    sys.modules[__name__].LightningTask = LightningTask
    model, trainer, _ = build(
        {"model": {"_target_": f"{__name__}.LightningTask"}, "trainer": dict(TRAINER, _target_="lightning.pytorch.Trainer")}
    )
    assert isinstance(model, L.LightningModule) and isinstance(trainer, L.Trainer)
    with pytest.raises(TypeError, match="mixes `pytorch_lightning` and `lightning.pytorch`"):
        build({"model": {"_target_": f"{__name__}.LightningTask"}, "trainer": dict(TRAINER)})


def test_seed_defaults_to_zero_without_a_warning(monkeypatch):
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build(config())
    assert os.environ["PL_GLOBAL_SEED"] == "0"
    assert os.environ["PL_SEED_WORKERS"] == "1"


def test_seed_comes_from_the_config_and_does_not_carry_over(monkeypatch):
    build(config(seed=42))
    first = torch.rand(1)
    build(config(seed=42))
    assert torch.equal(first, torch.rand(1))
    build(config())  # a later build in the same process starts again from 0
    assert os.environ["PL_GLOBAL_SEED"] == "0"


def test_seed_override_needs_no_seed_key_in_the_config():
    build(config(), "seed=7")
    assert os.environ["PL_GLOBAL_SEED"] == "7"


@pytest.mark.parametrize("seed", ["@base", "$@base + 0"])
def test_seed_may_be_a_reference_or_an_expression(monkeypatch, seed):
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    build(config(seed=seed, base=5))
    assert os.environ["PL_GLOBAL_SEED"] == "5"


def test_build_logs_a_banner(tmp_path, caplog):
    caplog.set_level("INFO", logger="lighter")
    build(config(trainer=dict(TRAINER, logger=True, default_root_dir=str(tmp_path))), "seed=3")
    banner = [record.getMessage() for record in caplog.records if record.name.startswith("lighter")]
    assert banner == [
        f"Lighter {lighter.__version__} | project: none | dict + 1 overrides | seed: 3",
        f"Logs: {tmp_path / 'lightning_logs' / 'version_0'}",
    ]


def test_seed_must_be_an_integer():
    with pytest.raises(TypeError, match="`seed` must be an integer"):
        build(config(seed="42"))


def test_python_dash_m_lighter_runs_the_cli():
    result = subprocess.run([sys.executable, "-m", "lighter", "--help"], capture_output=True, text=True, check=True)
    assert "usage: lighter" in result.stdout


# Project discovery

NETS = "import torch\nfrom pytorch_lightning import LightningModule\nclass Net(LightningModule):\n    pass\n"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project folder (`__lighter__.py`, no `__init__.py`) with its config in `configs/`, run from elsewhere."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "__lighter__.py").touch()
    (root / "toynets.py").write_text(NETS)
    trainer = "".join(f"  {k}: {v}\n" for k, v in TRAINER.items())
    (root / "configs" / "train.yaml").write_text(f"model:\n  _target_: project.toynets.Net\ntrainer:\n{trainer}")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    yield root
    drop()


def test_project_is_found_from_the_config_folder(project):
    model, _, _ = build(str(project / "configs" / "train.yaml"))
    assert type(model).__module__ == "project.toynets"
    assert find_project_root((str(project / "configs" / "train.yaml"),)) == project


def test_project_is_found_from_the_cwd_for_dict_inputs(project, monkeypatch):
    monkeypatch.chdir(project / "configs")
    model, _, _ = build(config(model={"_target_": "project.toynets.Net"}))
    assert type(model).__module__ == "project.toynets"


def test_configs_from_two_projects_are_rejected(project, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "__lighter__.py").touch()
    (other / "extra.yaml").write_text("seed: 1\n")
    with pytest.raises(ValueError, match="different projects"):
        build(str(project / "configs" / "train.yaml"), str(other / "extra.yaml"))


def test_relative_parent_paths_do_not_warn(project, monkeypatch):
    monkeypatch.chdir(project / "configs")
    (project / "sub").mkdir()
    monkeypatch.chdir(project / "sub")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_config("../configs/train.yaml").get("model::_target_") == "project.toynets.Net"


def test_target_without_project_prefix_suggests_it(project):
    with pytest.raises(TargetNotFoundError, match=r"did you mean 'project\.toynets\.Net'"):
        build(str(project / "configs" / "train.yaml"), "model::_target_=toynets.Net")


def test_missing_marker_suggests_creating_it(project):
    (project / "__lighter__.py").unlink()
    with pytest.raises(TargetNotFoundError, match=f"create an empty {re.escape(str(project / '__lighter__.py'))}"):
        build(str(project / "configs" / "train.yaml"))


def test_installed_packages_get_no_project_hint():
    with pytest.raises(TargetNotFoundError) as error:
        build(config(model={"_target_": "torch.nn.NotAModule"}))
    assert "project" not in str(error.value)


@pytest.mark.parametrize("value, raises", [("1", True), ("true", True), ("false", False), ("0", False)])
def test_lighter_debug_shows_the_full_error(monkeypatch, tmp_path, value, raises):
    from lighter.engine.runner import cli

    monkeypatch.setenv("LIGHTER_DEBUG", value)
    monkeypatch.setattr(sys, "argv", ["lighter", "fit", str(tmp_path / "missing.yaml")])
    with pytest.raises(FileNotFoundError if raises else SystemExit):
        cli()
