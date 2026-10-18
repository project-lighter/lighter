"""The examples run as the README and the docs show them.

The README cannot include files, so it shows copies of examples/quickstart. After changing those files,
refresh the copies with `UPDATE_GOLDEN=1 pytest tests/integration/test_examples.py`.
"""

import csv
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

from lighter import Runner
from lighter.engine.runner import cli

ROOT = Path(__file__).parents[2]
QUICKSTART = ROOT / "examples" / "quickstart"
TABULAR = ROOT / "projects" / "tabular_regression"
README_FILE = re.compile(r"(<!-- (examples/\S+) -->\n```\w*\n)(.*?)(```)", re.S)


@pytest.fixture
def fresh_project():
    """Each test imports its own `project` package."""
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    yield
    drop()


def lighter(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["lighter", *args])
    cli()


def test_quickstart_learns_and_takes_an_override_from_another_folder(tmp_path, monkeypatch, fresh_project):
    monkeypatch.chdir(tmp_path)
    config = os.path.relpath(QUICKSTART / "config.yaml")  # e.g. ../../home/you/lighter/examples/quickstart/config.yaml
    lighter(monkeypatch, "fit", config)
    lighter(monkeypatch, "fit", config, "model::lr=0.001")

    first, second = sorted((tmp_path / "lightning_logs").iterdir())
    assert (first.name, second.name) == ("version_0", "version_1")
    losses = [float(row["train_loss"]) for row in csv.DictReader((first / "metrics.csv").open()) if row["train_loss"]]
    assert losses[-1] < losses[0] / 10
    assert yaml.safe_load((first / "config.yaml").read_text())["model"]["lr"] == 0.01
    assert yaml.safe_load((second / "config.yaml").read_text())["model"]["lr"] == 0.001


def test_tabular_regression_fits_tests_and_predicts(tmp_path, monkeypatch, fresh_project):
    monkeypatch.chdir(tmp_path)
    config = str(TABULAR / "config.yaml")
    lighter(monkeypatch, "fit", config)
    checkpoint = str(next(tmp_path.glob("lightning_logs/version_0/checkpoints/*.ckpt")))

    [results] = Runner().run("test", [config], ckpt_path=checkpoint, verbose=False)
    assert results["test/metrics/MeanAbsoluteError/epoch"] < 0.01  # the known linear function is learned
    [batch] = Runner().run("predict", [config], ckpt_path=checkpoint)
    assert batch["id"][:2] == ["00200", "00201"]  # the held-out test samples, with their IDs
    assert (batch["prediction"] - batch["target"]).abs().max() < 0.01


def test_readme_shows_the_quickstart_files_as_they_are():
    readme = ROOT / "README.md"
    text = readme.read_text()
    if os.environ.get("UPDATE_GOLDEN"):
        text = README_FILE.sub(lambda m: m[1] + (ROOT / m[2]).read_text() + m[4], text)
        readme.write_text(text)
    shown = {path: code for _, path, code, _ in README_FILE.findall(text)}
    assert set(shown) == {f"examples/quickstart/{name}" for name in ("model.py", "data.py", "config.yaml")}
    for path, code in shown.items():
        assert code == (ROOT / path).read_text(), f"README.md shows an old {path}; run UPDATE_GOLDEN=1 pytest {__file__}"
