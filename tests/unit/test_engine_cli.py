"""Unit tests for the `lighter` command line (lighter/engine/runner.py:cli)."""

import subprocess
import sys
from unittest.mock import MagicMock, create_autospec

import pytest

import lighter
from lighter.engine.runner import cli

CONFIG = "trainer:\n  _target_: pytorch_lightning.Trainer\n  max_epochs: 10\nmodel:\n  _target_: torch.nn.Identity\n"


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return str(path)


@pytest.fixture
def built(monkeypatch):
    """Replace `build` with a mock that returns a mock trainer."""
    model, trainer, datamodule = MagicMock(), MagicMock(), MagicMock()
    build = MagicMock(return_value=(model, trainer, datamodule))
    monkeypatch.setattr("lighter.engine.runner.build", build)
    return build, model, trainer, datamodule


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["lighter", *args])
    try:
        cli()
    except SystemExit as exit:
        return exit.code
    return 0


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_cli_builds_then_runs_the_stage(monkeypatch, built, stage, config_file):
    build, model, trainer, datamodule = built
    assert run_cli(monkeypatch, stage, config_file, "trainer::max_epochs=5", "model::name=a=b") == 0
    build.assert_called_once_with(config_file, "trainer::max_epochs=5", "model::name=a=b")
    getattr(trainer, stage).assert_called_once_with(model, datamodule=datamodule)


def test_cli_passes_stage_flags(monkeypatch, built, config_file):
    _, model, trainer, datamodule = built
    trainer.test = create_autospec(lambda model, datamodule=None, ckpt_path=None, weights_only=None, verbose=None: None)
    run_cli(monkeypatch, "test", config_file, "--ckpt_path", "best.ckpt", "--no-weights_only", "--verbose")
    trainer.test.assert_called_once_with(model, datamodule=datamodule, ckpt_path="best.ckpt", weights_only=False, verbose=True)
    run_cli(monkeypatch, "predict", config_file, "--no-return_predictions")
    trainer.predict.assert_called_once_with(model, datamodule=datamodule, return_predictions=False)


@pytest.mark.parametrize("args", [[], ["fit"], ["train", "config.yaml"]])
def test_cli_rejects_bad_usage(monkeypatch, args):
    assert run_cli(monkeypatch, *args) == 2


def test_cli_version_and_help(monkeypatch, capsys):
    assert run_cli(monkeypatch, "--version") == 0
    assert capsys.readouterr().out == f"lighter {lighter.__version__}\n"
    assert run_cli(monkeypatch, "fit", "--help") == 0
    help_text = capsys.readouterr().out
    for syntax in ("key::sub=value", "=key={...}", "~key", "'model::lr=$@base_lr * 2'", "--print_config", "LIGHTER_DEBUG"):
        assert syntax in help_text


@pytest.mark.parametrize("flag", ["--print_config", "--print-config"])
def test_print_config_prints_the_composed_config_without_building(monkeypatch, capsys, built, config_file, flag):
    assert run_cli(monkeypatch, "fit", config_file, "trainer::max_epochs=3", flag) == 0
    assert capsys.readouterr().out == "seed: 0\n" + CONFIG.replace("max_epochs: 10", "max_epochs: 3")
    built[0].assert_not_called()


def test_missing_config_file_exits_with_code_2(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert run_cli(monkeypatch, "fit", "missing.yaml") == 2
    hint = f"Paths are relative to the current directory, {tmp_path}."
    assert capsys.readouterr().err == f"FileNotFoundError: Config file not found: missing.yaml\n  {hint}\n"


def test_build_errors_are_reported_without_a_traceback(monkeypatch, capsys, built, config_file):
    built[0].side_effect = TypeError("`trainer` must build a Trainer, got dict.")
    assert run_cli(monkeypatch, "fit", config_file) == 2
    assert capsys.readouterr().err == "TypeError: `trainer` must build a Trainer, got dict.\n"


def test_lighter_debug_shows_the_full_traceback(monkeypatch, built, config_file):
    built[0].side_effect = TypeError("bad config")
    monkeypatch.setenv("LIGHTER_DEBUG", "1")
    with pytest.raises(TypeError, match="bad config"):
        run_cli(monkeypatch, "fit", config_file)


def test_errors_inside_the_stage_propagate_unchanged(monkeypatch, built, config_file):
    error = RuntimeError("CUDA out of memory")
    error.__cause__ = ValueError("root cause")
    built[2].fit.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        run_cli(monkeypatch, "fit", config_file)
    assert raised.value is error and raised.value.__cause__ is error.__cause__


def test_cli_starts_without_torch_and_ignores_the_cwd(tmp_path, config_file):
    """`--print_config` and file errors do not import torch; `python -m lighter` does not import from the cwd."""
    script = (
        "import sys\nfrom lighter.engine.runner import cli\ntry:\n    cli()\nfinally:\n    assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script, "fit", config_file, "--print_config"], check=True, capture_output=True)
    missing = subprocess.run([sys.executable, "-c", script, "fit", "missing.yaml"], capture_output=True, text=True)
    assert missing.returncode == 2 and "Config file not found" in missing.stderr, missing.stderr
    (tmp_path / "yaml.py").write_text("raise SystemExit('imported from the cwd')\n")
    result = subprocess.run([sys.executable, "-m", "lighter", "fit", "--help"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0 and "usage: lighter fit" in result.stdout, result.stderr


def test_weights_only_needs_a_lightning_that_accepts_it(monkeypatch, capsys, built, config_file):
    _, _, trainer, _ = built
    trainer.fit = create_autospec(lambda model, datamodule=None, ckpt_path=None: None)  # Lightning before 2.6
    assert run_cli(monkeypatch, "fit", config_file, "--no-weights_only") == 2
    assert "--weights_only needs Lightning 2.6 or newer" in capsys.readouterr().err
    trainer.fit.assert_not_called()


def test_ckpt_path_best_is_rejected(monkeypatch, capsys, built, config_file):
    assert run_cli(monkeypatch, "test", config_file, "--ckpt_path", "best") == 2
    assert "--ckpt_path best works only after `fit` in the same process" in capsys.readouterr().err
