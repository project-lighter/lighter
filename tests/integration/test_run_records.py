"""Local records observe native execution without changing its scientific behavior."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader

from lighter import LighterModule
from lighter.engine.records import RunRecorder, atomic_write, describe, diff_records, list_records, read_record


class RecordTask(LightningModule):
    def __init__(self, lr=0.01, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.lr = lr
        self.fail = fail

    def training_step(self, batch, batch_idx):
        if self.fail:
            raise RuntimeError("scientific step failed")
        self.log("train/score", self.weight.detach(), on_epoch=True, logger=False, batch_size=1)
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("score", batch.float().mean(), logger=False, batch_size=len(batch))

    test_step = validation_step

    def predict_step(self, batch, batch_idx):
        return batch * 2

    def train_dataloader(self):
        return DataLoader([1, 2], batch_size=1)

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader
    predict_dataloader = train_dataloader

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def engine(tmp_path, callbacks, **kwargs):
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=callbacks,
        default_root_dir=str(tmp_path),
        **kwargs,
    )


def recorder(stage="fit", **kwargs):
    return RunRecorder(
        source={"model": {"_target_": "example.Task", "lr": 0.5}, "seed": 42},
        stage=stage,
        seed=42,
        requested_args=kwargs,
        inputs=[],
    )


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_record_does_not_replace_native_stage_result(tmp_path, stage):
    record = recorder(stage)
    trainer = engine(tmp_path, [record], max_epochs=1)
    result = getattr(trainer, stage)(RecordTask())
    saved = read_record(record.path)
    assert saved["schema_version"] == 1
    assert saved["stage"] == stage
    assert saved["status"] == "completed"
    assert saved["execution"]["world_size"] == 1
    assert saved["seed"] == {"value": 42, "workers": True}
    assert saved["requested"]["source"]["model"]["lr"] == 0.5
    assert Path(record.path).with_name("config.yaml").is_file()
    if stage == "predict":
        assert torch.equal(torch.cat(result), torch.tensor([2, 4]))
    elif stage == "fit":
        assert result is None
        assert saved["observed_start"]["optimizers"][0]["groups"][0]["settings"]["lr"] == 0.01
    else:
        # Native logger-disabled evaluation omits returned values but callback metrics remain.
        assert result == [{}]
        assert saved["metrics"]["score"] == 1.5


def test_fit_sanity_and_validation_do_not_complete_attempt(tmp_path):
    record = recorder()
    observations = []

    class Observe(Callback):
        def on_validation_end(self, trainer, pl_module):
            observations.append(read_record(record.path)["status"])

    engine(tmp_path, [record, Observe()], max_epochs=1).fit(RecordTask())
    assert observations == ["running", "running"]
    assert read_record(record.path)["status"] == "completed"


def test_failed_native_step_is_recorded_and_exception_preserved(tmp_path):
    record = recorder()
    with pytest.raises(RuntimeError, match="scientific step failed"):
        engine(tmp_path, [record], max_epochs=1).fit(RecordTask(fail=True))
    saved = read_record(record.path)
    assert saved["status"] == "failed"
    assert saved["error"] == {"type": "RuntimeError", "message": "scientific step failed"}


def test_observations_distinguish_restored_lr_and_progress_from_request(tmp_path):
    checkpoint = tmp_path / "native.ckpt"
    first = engine(tmp_path, [], max_epochs=1, limit_train_batches=1)
    first.fit(RecordTask(lr=0.01))
    first.save_checkpoint(checkpoint)
    record = recorder(ckpt_path=str(checkpoint))
    engine(tmp_path, [record], max_epochs=2, limit_train_batches=1).fit(RecordTask(lr=0.5), ckpt_path=checkpoint)
    saved = read_record(record.path)
    assert saved["requested"]["source"]["model"]["lr"] == 0.5
    assert saved["requested"]["stage_arguments"]["ckpt_path"] == str(checkpoint)
    assert saved["observed_start"]["optimizers"][0]["groups"][0]["settings"]["lr"] == 0.01
    assert saved["observed_start"]["global_step"] == 1
    assert saved["observed_end"]["global_step"] == 2
    assert saved["observed_start"]["checkpoint_path"] == str(checkpoint)


def test_unique_attempts_list_and_compare_without_recipe_execution(tmp_path):
    first, second = recorder("predict"), recorder("predict", return_predictions=False)
    engine(tmp_path, [first]).predict(RecordTask())
    engine(tmp_path, [second]).predict(RecordTask(), return_predictions=False)
    assert first.path != second.path
    assert len(list_records(tmp_path / "lighter_runs")) == 2
    changes = diff_records(first.path, second.path)
    assert any(
        change["path"] == "requested::stage_arguments::return_predictions" and change["after"] is False for change in changes
    )


def test_opaque_values_are_described_without_invoking_them():
    class Opaque:
        def __repr__(self):
            raise AssertionError("repr is user code")

    value = describe({"model": Opaque()})
    assert value["model"]["replayable"] is False
    assert value["model"]["__opaque__"].endswith("Opaque")
    json.dumps(value, allow_nan=False)


def test_atomic_record_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "record.json"
    atomic_write(path, '{"status": "running"}')
    with patch.object(Path, "replace", side_effect=OSError("disk error")), pytest.raises(OSError, match="disk error"):
        atomic_write(path, '{"status": "completed"}')
    assert json.loads(path.read_text())["status"] == "running"
    assert list(tmp_path.iterdir()) == [path]


def test_record_write_failure_does_not_replace_native_exception(tmp_path):
    record = recorder()
    with (
        patch("lighter.engine.records.atomic_write", side_effect=OSError("storage unavailable")),
        pytest.warns(RuntimeWarning),
    ):
        with pytest.raises(RuntimeError, match="scientific step failed"):
            engine(tmp_path, [record], max_epochs=1).fit(RecordTask(fail=True))
    assert "storage unavailable" in record.last_error


def test_arbitrary_native_optimizer_group_metadata_does_not_break_recording(tmp_path):
    class CustomGroup(RecordTask):
        def configure_optimizers(self):
            optimizer = super().configure_optimizers()
            optimizer.param_groups[0][7] = "custom metadata"
            return optimizer

    record = recorder()
    model = CustomGroup()
    engine(tmp_path, [record], max_steps=1).fit(model)
    group = read_record(record.path)["observed_start"]["optimizers"][0]["groups"][0]
    assert group["settings"]["lr"] == 0.01
    assert group["non_string_settings"] == [{"key": 7, "value": "custom metadata"}]
    assert model.weight.item() == pytest.approx(0.98)


def test_runner_records_are_scoped_when_native_trainer_callbacks_are_replaced(tmp_path):
    from lighter import Runner

    class WithCallback(RecordTask):
        def configure_callbacks(self):
            return [Callback()]

    trainer = engine(tmp_path, [])
    task = WithCallback()
    runner = Runner()
    original_callbacks = list(trainer.callbacks)
    first = runner.run("predict", [{"model": task, "trainer": trainer}])
    first_path = runner.last_run_path
    first_contents = first_path.read_text()
    assert all(not isinstance(callback, RunRecorder) for callback in trainer.callbacks)
    second = runner.run("test", [{"model": task, "trainer": trainer}])
    assert runner.last_run_path != first_path
    assert first_path.read_text() == first_contents
    assert read_record(runner.last_run_path)["stage"] == "test"
    assert all(callback in trainer.callbacks for callback in original_callbacks)
    assert all(not isinstance(callback, RunRecorder) for callback in trainer.callbacks)
    assert torch.equal(torch.cat(first), torch.tensor([2, 4]))
    assert second == [{}]


def test_runner_opt_out_and_failure_record_cleanup(tmp_path):
    from lighter import Runner

    trainer = engine(tmp_path, [], max_epochs=1)
    runner = Runner()
    runner.run("predict", [{"model": RecordTask(), "trainer": trainer, "run": False}])
    assert runner.last_run_path is None
    assert not (tmp_path / "lighter_runs").exists()
    with pytest.raises(RuntimeError, match="scientific step failed"):
        runner.run("fit", [{"model": RecordTask(fail=True), "trainer": trainer}])
    assert read_record(runner.last_run_path)["status"] == "failed"
    assert all(not isinstance(callback, RunRecorder) for callback in trainer.callbacks)


class OpaqueInputTask(LighterModule):
    def training_step(self, batch, batch_idx):
        return self.network(batch).square().mean()


def test_runner_checkpoint_metadata_does_not_copy_or_consume_opaque_inputs(tmp_path):
    from lighter import Runner

    network = torch.nn.Linear(1, 1, bias=False)
    network.weight.data.fill_(1.0)
    trainer = engine(tmp_path, [], max_steps=1)
    recipe = {
        "trainer": trainer,
        "model": {
            "_target_": f"{__name__}.OpaqueInputTask",
            "network": network,
            "optimizer": {"_target_": "torch.optim.SGD", "params": network.parameters(), "lr": 0.01},
        },
    }
    runner = Runner()
    runner.run("fit", [recipe], train_dataloaders=DataLoader([torch.ones(1)], batch_size=1))
    assert network.weight.item() == pytest.approx(0.98)
    metadata = trainer.lightning_module.hparams["config"]["model"]["optimizer"]["params"]
    assert metadata["replayable"] is False
    assert metadata["__opaque__"] == "builtins.generator"
    assert read_record(runner.last_run_path)["source_contains_opaque_values"] is True


def test_epoch_progress_is_visible_before_fit_completion(tmp_path):
    record = recorder()
    progress = []

    class Observe(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            saved = read_record(record.path)
            progress.append((saved["status"], saved["progress"]["global_step"]))

    engine(tmp_path, [record, Observe()], max_epochs=2).fit(RecordTask())
    assert progress == [("running", 2), ("running", 4)]
    assert read_record(record.path)["status"] == "completed"


def test_prepare_preserves_attempt_identity_without_publishing(tmp_path):
    import pickle

    record = recorder()
    trainer = engine(tmp_path, [], max_epochs=1)
    record.prepare(trainer)
    assert record.path is not None and not record.path.exists()
    spawned = pickle.loads(pickle.dumps(record))
    spawned.setup(trainer, RecordTask(), "fit")
    assert spawned.path == record.path
    assert read_record(record.path)["attempt_id"] == record.attempt_id


@pytest.mark.parametrize("installed_in_ignored_environment", [False, True])
def test_package_git_provenance_requires_a_tracked_module(tmp_path, monkeypatch, installed_in_ignored_environment):
    import shutil
    import subprocess
    import sys
    from types import ModuleType

    from lighter.engine.records import _environment

    if shutil.which("git") is None:
        pytest.skip("Git unavailable")

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("environment/\n")
    tracked = tmp_path / "src/lighter/__init__.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text('__version__ = "example"\n')
    git("add", ".gitignore", "src")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
    installed = tmp_path / "environment/site-packages/lighter/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text(tracked.read_text())
    module = ModuleType("lighter")
    module.__file__ = str(installed if installed_in_ignored_environment else tracked)
    monkeypatch.setitem(sys.modules, "lighter", module)
    evidence = _environment()["packages"]["lighter"]["source_git"]
    if installed_in_ignored_environment:
        assert evidence is None, "A surrounding application's Git commit is not the installed package's source revision"
    else:
        assert evidence["commit"]
        assert Path(evidence["root"]).resolve() == tmp_path.resolve()


@pytest.fixture
def record_construction(tmp_path, monkeypatch):
    """Count real construction, replacing only project discovery and native stage execution."""
    from lighter import Runner
    from lighter.engine.runner import ProjectImporter

    events = []
    for name, cls in (
        ("network", torch.nn.Linear),
        ("model", LighterModule),
        ("trainer", Trainer),
        ("data", LightningDataModule),
        ("optimizer", torch.optim.SGD),
    ):
        original = cls.__init__

        def counted(self, *args, _name=name, _original=original, **kwargs):
            events.append(_name)
            _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", counted)

    original_save = Runner._save_config

    def save(self, *args, **kwargs):
        events.append("snapshot")
        return original_save(self, *args, **kwargs)

    def stage(self, stage, model, trainer, datamodule, **kwargs):
        events.append("stage")
        return [callback.options for callback in trainer.callbacks if isinstance(callback, RunRecorder)]

    monkeypatch.setattr(ProjectImporter, "auto_discover_and_import", lambda: events.append("project"))
    monkeypatch.setattr(Runner, "_execute", stage)
    monkeypatch.setattr(Runner, "_save_config", save)
    config = {
        "model": {
            "_target_": f"{__name__}.OpaqueInputTask",
            "network": {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
            "optimizer": {"_target_": "torch.optim.SGD", "params": "$@model::network.parameters()", "lr": 0.05},
        },
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
        "data": {"_target_": "pytorch_lightning.LightningDataModule"},
    }
    return Runner(), config, events


@pytest.mark.parametrize("options", [None, True, 1, [], "@options", "%options", "$dict()"])
def test_literal_record_envelope_fails_before_construction(record_construction, options):
    runner, config, events = record_construction
    config["run"] = options
    with pytest.raises(TypeError, match="run must be false or a mapping of experiment record options"):
        runner.run("fit", [config])
    assert events == []
    assert runner.last_run_path is None
    assert list(Path(config["trainer"]["default_root_dir"]).rglob("config.yaml")) == []


@pytest.mark.parametrize("options", [{"unknown": "@missing"}, {"_disabled_": True}])
def test_literal_record_unknown_key_fails_before_construction(record_construction, options):
    runner, config, events = record_construction
    config["run"] = options
    with pytest.raises(ValueError, match="Unknown run record options"):
        runner.run("fit", [config])
    assert events == []


@pytest.mark.parametrize("value", [None, "", False, 0, 0.5, [], ["$1 / 0"], {}, ()])
def test_literal_record_value_fails_before_construction(record_construction, value):
    runner, config, events = record_construction
    config["run"] = {"parent_attempt_id": value}
    with pytest.raises(ValueError, match="run::parent_attempt_id must be a nonempty literal string"):
        runner.run("fit", [config])
    assert events == []
    assert list(Path(config["trainer"]["default_root_dir"]).rglob("config.yaml")) == []


@pytest.mark.parametrize("invalid", [{"seed": True}, {"args": {"unknown": {}}}])
def test_literal_record_check_preserves_seed_and_stage_error_precedence(record_construction, invalid):
    runner, config, events = record_construction
    config.update(invalid, run=None)
    expected = "seed must be an integer" if "seed" in invalid else "Unknown stage"
    with pytest.raises(ValueError, match=expected):
        runner.run("fit", [config])
    assert events == []


@pytest.mark.parametrize("options", [None, False, {}, {"name": " "}, {"name": "trial", "experiment_id": "series"}])
def test_literal_record_valid_options_reach_native_stage_boundary(record_construction, options):
    runner, config, events = record_construction
    if options is not None:  # None here denotes an omitted source section, not a null option.
        config["run"] = options
    result = runner.run("fit", [config])
    assert result == ([] if options is False else [options or {}])
    assert events == ["project", "network", "model", "trainer", "data", "snapshot", "stage"]


@pytest.mark.parametrize("kind", ["reference", "copy", "expression", "component", "whole"])
@pytest.mark.parametrize("value", ["trial", None])
def test_dynamic_record_options_keep_normal_resolution_and_final_validation(record_construction, kind, value):
    runner, config, events = record_construction

    def dynamic():
        events.append("dynamic")
        return {"name": value} if kind == "whole" else value

    component = {"_target_": dynamic}
    config["label"] = component
    definitions = {"reference": "@label", "copy": "%label", "expression": "$@label", "component": component}
    config["run"] = component if kind == "whole" else {"name": definitions[kind]}
    if value is None and kind not in {"copy", "component"}:
        with pytest.raises(ValueError, match="run::name must be a nonempty literal string"):
            runner.run("fit", [config])
        assert events == ["project", "network", "model", "trainer", "data", "snapshot", "dynamic"]
    else:
        # A nested component returning None is pruned by Sparkwheel; references
        # and expressions returning None remain values for final validation.
        expected = {} if value is None else {"name": "trial"}
        assert runner.run("fit", [config]) == [expected]
        assert events == ["project", "network", "model", "trainer", "data", "snapshot", "dynamic", "stage"]


@pytest.mark.parametrize("resolved", [None, False, 0, [], {}, [("name", "trial")]])
def test_dynamic_record_component_preserves_existing_option_normalization(record_construction, resolved):
    runner, config, events = record_construction

    def dynamic():
        events.append("dynamic")
        return resolved

    config["run"] = {"_target_": dynamic}
    expected = {"name": "trial"} if resolved else {}
    assert runner.run("fit", [config]) == [expected]
    assert events == ["project", "network", "model", "trainer", "data", "snapshot", "dynamic", "stage"]
    direct = RunRecorder(source={}, stage="fit", seed=0, requested_args={}, inputs=[], options=resolved)
    assert direct.options == expected


def test_literal_record_check_does_not_inspect_opaque_values_or_subclasses():
    from lighter.engine.records import _validate_literal_run_options

    def forbidden(*args, **kwargs):
        raise AssertionError("Early validation must not invoke opaque user methods")

    class Opaque:
        __bool__ = __iter__ = __repr__ = forbidden

    class CustomString(str):
        startswith = __bool__ = forbidden

    class CustomMapping(dict):
        __contains__ = __iter__ = keys = items = forbidden

    for value in (Opaque(), CustomString("dynamic"), CustomMapping()):
        _validate_literal_run_options({"name": value})
    _validate_literal_run_options(CustomMapping(name="trial"))

    class CustomKey(str):
        pass

    options = {CustomKey("name"): "trial"}
    nested = {CustomKey("_target_"): "unused"}
    CustomKey.__hash__ = CustomKey.__eq__ = forbidden
    _validate_literal_run_options(options)
    _validate_literal_run_options({"name": nested})
