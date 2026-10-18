"""docs/recipes.md works as written: every `lighter` command on it runs, in order, in a copy of
projects/tabular_regression, and the files it includes from projects/tabular_regression/recipes/ run too.

The GPU command is covered by tests/integration/test_ddp.py with two CPU processes, and the W&B command runs
only when wandb is installed.
"""

import csv
import importlib.util
import re
import runpy
import shlex
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

import lighter
from lighter.engine.runner import cli, load_config

ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "recipes.md"
SHELL = re.compile(r"^```bash\n(.*?)^```", re.M | re.S)
LOOP = re.compile(r"for (\w+) in ([^;]+); do (.+); done")


def page_commands() -> list[list[str]]:
    """The `lighter` commands in the page's shell blocks, with `for` loops unrolled."""
    found = []
    for block in SHELL.findall(PAGE.read_text()):
        for line in block.splitlines():
            if loop := LOOP.fullmatch(line.strip()):
                name, values, command = loop.groups()
                found += [shlex.split(command.replace(f"${name}", value)) for value in values.split()]
            elif line.startswith("lighter "):
                found.append(shlex.split(line, comments=True))
    return found


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A copy of projects/tabular_regression as the current folder."""
    folder = tmp_path / "tabular_regression"
    shutil.copytree(ROOT / "projects" / "tabular_regression", folder, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.chdir(folder)
    monkeypatch.setenv("WANDB_MODE", "offline")
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    yield folder
    drop()


def runs(folder: Path) -> list[tuple[dict, Path]]:
    """Each run's saved config and folder, in the order they ran."""
    found = sorted((folder / "lightning_logs").iterdir(), key=lambda run: int(run.name.split("_")[1]))
    return [(yaml.safe_load((run / "config.yaml").read_text()), run) for run in found]


def epochs(run: Path) -> set[int]:
    return {int(row["epoch"]) for row in csv.DictReader((run / "metrics.csv").open()) if row["epoch"]}


def test_every_command_on_the_page_runs_and_does_what_it_says(project, monkeypatch, capsys):
    commands = page_commands()
    assert len(commands) == 12
    skip = ["trainer::accelerator=gpu", *([] if importlib.util.find_spec("wandb") else ["recipes/wandb.yaml"])]
    for command in commands:
        if not any(word in command for word in skip):
            monkeypatch.setattr(sys, "argv", command)
            cli()
    done = runs(project)

    # Try a variant: the stacked file replaced the network.
    assert done[0][0]["model"]["network"]["_target_"] == "torch.nn.Sequential"
    # Resume training: the second run went on from the first one's last checkpoint.
    assert epochs(done[1][1]) == set(range(20)) and epochs(done[2][1]) == set(range(20, 30))
    # Repeat a run: the saved config of the first run gave the same run again.
    assert done[3][0] == done[0][0]
    assert (done[3][1] / "metrics.csv").read_text() == (done[0][1] / "metrics.csv").read_text()
    # Seed a run.
    assert done[4][0]["seed"] == 7
    # Freeze layers: the first layer is as it was built (with the same seed), the last one trained.
    built, _, _ = lighter.build("config.yaml", "recipes/mlp.yaml")
    trained = torch.load(next(done[5][1].glob("checkpoints/*.ckpt")), weights_only=True)["state_dict"]
    assert torch.equal(trained["network.0.weight"], built.network[0].weight.detach())
    assert not torch.equal(trained["network.2.weight"], built.network[2].weight.detach())
    # Write predictions to a file: one row per test sample, with its ID.
    rows = list(csv.DictReader((project / "predictions.csv").open()))
    assert [row["id"] for row in rows] == [f"{i:05d}" for i in range(200, 216)]
    assert max(abs(float(row["prediction"]) - float(row["target"])) for row in rows) < 0.01
    # Sweep hyperparameters with a shell loop (after the folder of the predict run).
    assert [config["model"]["optimizer"]["lr"] for config, _ in done[7:10]] == [0.3, 0.1, 0.03]

    # Use a trained model in a notebook.
    capsys.readouterr()
    runpy.run_path("recipes/load_model.py")
    assert abs(float(re.search(r"\[\[([-\d.]+)\]\]", capsys.readouterr().out)[1]) - 4.5) < 0.05


def test_optuna_sweep_runs_trials_with_the_suggested_learning_rates(project, monkeypatch, capsys):
    """Optuna is not a dependency, so a stand-in suggests two values and keeps the best."""

    class Trial:
        def __init__(self, lr):
            self.params = {"lr": lr}

        def suggest_float(self, name, low, high, log=False):
            assert low <= self.params[name] <= high and log
            return self.params[name]

    class Study:
        def optimize(self, objective, n_trials):
            assert n_trials > 1
            results = {objective(Trial(lr)): lr for lr in (0.003, 0.1)}
            self.best_params = {"lr": results[min(results)]}

    optuna = types.SimpleNamespace(Trial=Trial, create_study=lambda direction: Study())
    monkeypatch.setitem(sys.modules, "optuna", optuna)
    runpy.run_path("recipes/optuna_sweep.py", run_name="__main__")
    assert capsys.readouterr().out.splitlines()[-1] == "{'lr': 0.1}"  # 0.003 learns too slowly in 20 epochs
    assert [config["model"]["optimizer"]["lr"] for config, _ in runs(project)] == [0.003, 0.1]


def test_wandb_sweep_command_runs_with_the_overrides_the_agent_passes(project, monkeypatch, capsys):
    sweep = yaml.safe_load((project / "recipes" / "wandb_sweep.yaml").read_text())
    config = load_config("config.yaml")
    overrides = [f"{name}=0.01" for name in sweep["parameters"]]
    assert all(config.get(name) is not None for name in sweep["parameters"])
    command = [part for item in sweep["command"] for part in (overrides if item == "${args_no_hyphens}" else [item])]
    monkeypatch.setattr(sys, "argv", [*command, "--print_config"])
    cli()
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["model"]["optimizer"]["lr"] == 0.01
    assert printed["trainer"]["logger"]["_target_"] == "pytorch_lightning.loggers.WandbLogger"
