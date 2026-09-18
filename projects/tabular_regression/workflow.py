"""Run and verify the documented public CLI workflow in separate processes."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from task import PREDICTION_IDS


def run_workflow(output_dir: Path) -> dict:
    project_dir = Path(__file__).resolve().parent
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    override = f"output_dir={json.dumps(str(output_dir))}"
    commands = []
    inspection_commands = []

    def inspect_json(label, *arguments):
        command = [sys.executable, "-m", "lighter", *arguments]
        inspection_commands.append(command)
        result = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        (output_dir / f"{label}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        return value

    source = inspect_json("source", "inspect", "config.yaml", override, "--json")
    changed = inspect_json("changed-source", "inspect", "config.yaml", override, "model::optimizer::lr=0.05", "--json")
    assert source["model"]["optimizer"]["lr"] == 0.01
    assert changed["model"]["optimizer"]["lr"] == 0.05
    record_root = output_dir / "lighter_runs"
    prior_ids = {path.parent.name for path in record_root.glob("*/record.json")}

    def execute(label, stage, *arguments):
        command = [sys.executable, "-m", "lighter", stage, "config.yaml", override, *arguments]
        commands.append(command)
        result = subprocess.run(command, cwd=project_dir, capture_output=True, text=True)
        (output_dir / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {output_dir / f'{label}.log'}")

    # Exercise a real nested override: the recipe default is 0.01.
    execute("fit", "fit", "model::optimizer::lr=0.05")
    checkpoint = output_dir / "checkpoints" / "last.ckpt"
    initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial_step = initial["global_step"]
    assert initial_step == 9, f"Expected 3 epochs × 3 batches; got {initial_step}"
    assert initial["hyper_parameters"]["config"]["model"]["optimizer"]["lr"] == 0.05
    assert initial["optimizer_states"][0]["param_groups"][0]["lr"] == 0.05

    # Always name a concrete checkpoint in a fresh CLI process.
    execute("test", "test", "--ckpt-path", str(checkpoint), "--no-verbose")
    execute("predict", "predict", "--ckpt-path", str(checkpoint), "--no-return-predictions")
    with (output_dir / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["id"] for row in rows]
    assert ids == list(PREDICTION_IDS), f"Missing, duplicated or changed sample IDs: {ids!r}"
    # Independent native model oracle: the exported value must correspond to
    # this exact checkpoint and this row's sample, not merely be finite.
    native = torch.nn.Linear(2, 1)
    native.load_state_dict(
        {key.removeprefix("network."): value for key, value in initial["state_dict"].items() if key.startswith("network.")}
    )
    with torch.no_grad():
        for index, row in enumerate(rows):
            sample = 80 + index  # The documented held-out population.
            features = torch.tensor([math.sin(sample), math.cos(sample * 1.3)], dtype=torch.float32)
            expected_prediction = native(features).item()
            expected_target = 2 * math.sin(sample) - 3 * math.cos(sample * 1.3) + 0.5
            assert math.isclose(float(row["prediction"]), expected_prediction, rel_tol=1e-6, abs_tol=1e-6)
            assert math.isclose(float(row["target"]), expected_target, rel_tol=1e-6, abs_tol=1e-6)

    execute("resume", "fit", "--ckpt-path", str(checkpoint), "trainer::max_epochs=5")
    continued = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert continued["global_step"] == 15
    assert continued["epoch"] == 4
    # Resume restores saved optimizer settings despite the recipe default 0.01.
    assert continued["optimizer_states"][0]["param_groups"][0]["lr"] == 0.05
    assert continued["optimizer_states"][0]["state"], "SGD momentum state must survive continuation"
    records = inspect_json("attempts", "runs", "list", str(record_root), "--json")
    records = [record for record in records if record["attempt_id"] not in prior_ids]
    assert sorted(record["stage"] for record in records) == ["fit", "fit", "predict", "test"]
    assert all(record["status"] == "completed" for record in records)
    fits = sorted(
        (record for record in records if record["stage"] == "fit"), key=lambda item: item["observed_start"]["global_step"]
    )
    first, resumed = fits
    assert first["observed_start"]["global_step"] == 0
    assert resumed["observed_start"]["global_step"] == initial_step
    assert resumed["observed_end"]["global_step"] == 15
    assert resumed["requested"]["source"]["model"]["optimizer"]["lr"] == 0.01
    assert resumed["observed_start"]["optimizers"][0]["groups"][0]["settings"]["lr"] == 0.05
    selected = inspect_json("selected-attempt", "runs", "show", str(record_root / first["attempt_id"]))
    assert selected == first
    changes = inspect_json(
        "attempt-diff", "runs", "diff", str(record_root / first["attempt_id"]), str(record_root / resumed["attempt_id"])
    )
    assert any(
        change["path"] == "requested::source::trainer::max_epochs" and change["before"] == 3 and change["after"] == 5
        for change in changes
    )
    summary = {
        "initial_step": initial_step,
        "continued_step": continued["global_step"],
        "prediction_ids": ids,
        "commands": commands,
        "inspection_commands": inspection_commands,
        "attempt_ids": [record["attempt_id"] for record in records],
    }
    (output_dir / "workflow.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tabular_regression"))
    options = parser.parse_args()
    print(json.dumps(run_workflow(options.output_dir), indent=2, ensure_ascii=False))
