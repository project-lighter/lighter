"""Real CPU/Gloo controls for construction, records and exact prediction export."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed

pytestmark = pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch Gloo backend unavailable")


def execute_fixture(name, arguments, output):
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), environment.get("PYTHONPATH", "")])
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, str(root / "tests/fixtures" / name), *arguments, str(output)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "result.json").read_text())


def test_managed_training_matches_native_and_analytic_updates_on_two_ranks(tmp_path):
    result = execute_fixture("distributed_training.py", [], tmp_path)
    assert result["passed"] and result["world_size"] == 2
    assert result["native_torch_optimizer_and_analytic_controls_agree"]


@pytest.mark.parametrize("kind", ["native", "lighter"])
@pytest.mark.parametrize("policy, expected_rows", [("auto", 5), ("explicit-padding", 6)])
def test_distributed_prediction_preserves_native_sampling_and_exact_rows(tmp_path, kind, policy, expected_rows):
    result = execute_fixture("distributed_prediction.py", [kind, policy], tmp_path)
    assert result["passed"] and result["emitted_rows"] == expected_rows
