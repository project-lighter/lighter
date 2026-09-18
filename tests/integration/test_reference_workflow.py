"""The canonical public example completes across independent CLI processes."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_reference_cli_workflow(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    # subprocesses inherit the tested checkout, even when cwd becomes the example.
    environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), environment.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, str(root / "projects/tabular_regression/workflow.py"), "--output-dir", str(tmp_path / "experiment")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "experiment/workflow.json").read_text())
    assert summary["initial_step"] == 9
    assert summary["continued_step"] == 15
    assert summary["prediction_ids"][:3] == ["00001", "NA", "NULL"]
    assert len(summary["prediction_ids"]) == len(set(summary["prediction_ids"])) == 7
    assert [command[3] for command in summary["commands"]] == ["fit", "test", "predict", "fit"]
