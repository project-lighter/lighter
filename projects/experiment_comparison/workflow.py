"""Automate the ordinary commands in fresh processes; keep this parent lightweight."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def check_controls(options):
    result = {"data_manifest": identity(options.data_manifest)}
    if bool(options.initial_state) != bool(options.order_manifest):
        raise ValueError("Supply both initial-state and order-manifest controls, or neither")
    if options.initial_state:
        sidecar = options.initial_state.parent / "controls.json"
        controls = json.loads(sidecar.read_text())
        for key, path in (("initial_state", options.initial_state), ("batch_order", options.order_manifest)):
            if digest(path) != controls[key]["sha256"]:
                raise ValueError(f"Control artifact digest mismatch: {key}")
            result[key] = identity(path)
        if controls["data_manifest"]["sha256"] != result["data_manifest"]["sha256"] or controls["seed"] != options.seed:
            raise ValueError("Controls reference a different data manifest or seed")
        order = json.loads(options.order_manifest.read_text())
        if order["batch_size"] != options.batch_size:
            raise ValueError("Control batch size mismatch")
        if order["data_manifest_sha256"] != result["data_manifest"]["sha256"]:
            raise ValueError("Order references a different data manifest")
        result["controls"] = identity(sidecar)
        result["case_revision"] = controls["case"]["revision"]
    return result


def check_checkpoint(path, controls):
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    if digest(path) != metadata["sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    for key in ("data_manifest", "initial_state", "batch_order"):
        if key in controls and controls[key]["sha256"] != metadata["identities"][key]["sha256"]:
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("lighter", "native"), default="lighter")
    parser.add_argument("--data-manifest", type=Path, default=Path("data/data.json"))
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--order-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--accelerator", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true")
    parser.add_argument("--evaluation", choices=("validation", "final"), default="validation")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluate-from", type=Path)
    modes.add_argument("--continue-from", type=Path)
    parser.add_argument("--parent-attempt-id")
    parser.add_argument("--trace-updates", action="store_true")
    return parser


def run(options):
    if options.evaluate_from and options.evaluation != "final":
        raise ValueError("Evaluate-only requires --evaluation final")
    if options.evaluation == "final" and not options.evaluate_from:
        raise ValueError("Final evaluation requires an explicit selected checkpoint; it never fits")
    if options.continue_from and not options.parent_attempt_id:
        raise ValueError("Continuation requires explicit parent-attempt-id")
    if options.max_epochs < 1 or options.lr <= 0:
        raise ValueError("max-epochs and learning rate must be positive")
    for name in ("data_manifest", "initial_state", "order_manifest", "evaluate_from", "continue_from", "output_dir"):
        if getattr(options, name) is not None:
            setattr(options, name, getattr(options, name).resolve())
    controls = check_controls(options)
    checkpoint_input = options.evaluate_from or options.continue_from
    parent_metadata = check_checkpoint(checkpoint_input, controls) if checkpoint_input else None
    if options.continue_from and options.parent_attempt_id != parent_metadata["parent_attempt_id"]:
        raise ValueError("Continuation parent-attempt-id does not identify the checkpoint creator")
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parent
    started = time.monotonic()
    mode = "evaluate" if options.evaluate_from else "continue" if options.continue_from else "fit"
    summary = {
        "schema_version": 1,
        "backend": options.backend,
        "mode": mode,
        "seed": options.seed,
        "requested_lr": options.lr,
        "requested_accelerator": options.accelerator,
        "requested_precision": options.precision,
        "max_epochs": options.max_epochs,
        "case_revision": controls.get("case_revision"),
        "identities": controls,
        "initial_validation": None,
        "prediction_population": "test" if mode == "evaluate" else None,
        "checkpoint_selection_scope": "source_attempt" if mode == "evaluate" else "current_attempt",
        "stages": [],
        "checkpoints": {},
        "metrics": {},
        "artifacts": {},
    }
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")

    def execute(name, stage, checkpoint=None):
        directory = output / name
        if options.backend == "lighter":
            overrides = {
                "output_dir": str(directory),
                "data_manifest": str(options.data_manifest),
                "initial_state": None if checkpoint or options.initial_state is None else str(options.initial_state),
                "order_manifest": str(options.order_manifest) if options.order_manifest else None,
                "seed": options.seed,
                "batch_size": options.batch_size,
                "learning_rate": options.lr,
                "trainer::max_epochs": options.max_epochs,
                "trainer::accelerator": options.accelerator,
                "trainer::precision": options.precision,
                "trace_updates": options.trace_updates,
            }
            if options.parent_attempt_id is not None:
                overrides["run::parent_attempt_id"] = options.parent_attempt_id
            command = [sys.executable, "-m", "lighter", stage, "config.yaml"]
            command += [f"{key}={json.dumps(value)}" for key, value in overrides.items()]
            if checkpoint:
                command += ["--ckpt-path", str(checkpoint)]
            if stage == "predict":
                command += ["--no-return-predictions"]
            if stage in ("validate", "test"):
                command += ["--no-verbose"]
        else:
            command = [
                sys.executable,
                str(project / "native.py"),
                stage,
                "--data-manifest",
                str(options.data_manifest),
                "--output-dir",
                str(directory),
                "--seed",
                str(options.seed),
                "--lr",
                str(options.lr),
                "--max-epochs",
                str(options.max_epochs),
                "--batch-size",
                str(options.batch_size),
                "--accelerator",
                options.accelerator,
                "--precision",
                options.precision,
            ]
            for flag, value in (
                ("--initial-state", options.initial_state),
                ("--order-manifest", options.order_manifest),
                ("--ckpt-path", checkpoint),
                ("--parent-attempt-id", options.parent_attempt_id),
            ):
                if value is not None:
                    command += [flag, str(value)]
            if options.trace_updates:
                command += ["--trace-updates"]
        stage_started = time.monotonic()
        with (output / f"{name}.stdout.log").open("w") as stdout, (output / f"{name}.stderr.log").open("w") as stderr:
            completed = subprocess.run(command, cwd=project, env=environment, stdout=stdout, stderr=stderr, timeout=600)
        record_paths = (
            list((directory / "lighter_runs").glob("*/record.json"))
            if options.backend == "lighter"
            else [directory / "native-attempt.json"]
        )
        records = [json.loads(path.read_text()) for path in record_paths if path.is_file()]
        record = records[0] if len(records) == 1 else {}
        runtime = directory / "runtime.json"
        stage_result = {
            "name": name,
            "stage": stage,
            "argv": command,
            "cwd": str(project),
            "exit_code": completed.returncode,
            "output_dir": name,
            "attempt_id": record.get("attempt_id"),
            "parent_attempt_id": options.parent_attempt_id,
            "elapsed_seconds": time.monotonic() - stage_started,
            "package_imports": identity(runtime) if runtime.exists() else None,
            "output_digests": {str(path.relative_to(output)): digest(path) for path in directory.rglob("*") if path.is_file()},
        }
        summary["stages"].append(stage_result)
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "workflow.json", summary)
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit {completed.returncode}; see preserved logs in {output}")
        if not record.get("attempt_id") or not runtime.exists():
            raise ValueError(f"{name} did not produce its attempt/runtime evidence")
        execution = json.loads(runtime.read_text())["execution"]
        if execution["device"].split(":")[0] != options.accelerator or execution["precision"] != options.precision:
            raise ValueError(f"{name} used a different device or precision than requested: {execution}")
        for filename in (
            "updates.pt",
            "observed_initial_state.pt",
            "observed_initial_optimizer.pt",
            "observed_start.json",
            "observed_end.json",
            "batches.jsonl",
            "epochs.csv",
            "validation.json",
            "test.json",
            "predictions.csv",
        ):
            path = directory / filename
            if path.exists():
                entry = {"path": str(path.relative_to(output)), "sha256": digest(path)}
                summary["artifacts"][f"{name}/{filename}"] = entry
                alias = {
                    "updates.pt": "updates",
                    "observed_initial_state.pt": "observed_initial_state",
                    "observed_initial_optimizer.pt": "observed_initial_optimizer",
                    "observed_start.json": "observed_start",
                    "epochs.csv": "epochs",
                    "validation.json": "validation",
                    "predictions.csv": "predictions",
                }.get(filename)
                if alias:
                    summary["artifacts"][alias] = entry
        return directory, record

    def metric(value):
        return {
            key: value[key] for key in ("count", "loss_sum", "loss", "correct", "accuracy", "confusion", "predicted_classes")
        } | {"ce": value["loss"]}

    if mode == "fit":
        # Source inspection deliberately does not execute model expressions.
        for name, extras in (("source", []), ("changed-source", ["learning_rate=0.1"])):
            command = [sys.executable, "-m", "lighter", "inspect", "config.yaml", *extras, "--json"]
            completed = subprocess.run(
                command, cwd=project, env=environment, capture_output=True, text=True, check=True, timeout=60
            )
            path = output / f"{name}.json"
            path.write_text(completed.stdout)
            summary["artifacts"][name] = {"path": path.name, "sha256": digest(path), "argv": command}

    if mode == "evaluate":
        summary["checkpoints"]["best"] = {**parent_metadata, "path": str(checkpoint_input)}
        summary["metrics"]["val"] = parent_metadata["metrics"]
        test_dir, _ = execute("test", "test", checkpoint_input)
        execute("predict", "predict", checkpoint_input)
        summary["metrics"]["test"] = metric(json.loads((test_dir / "test.json").read_text()))
    else:
        if mode == "fit":
            initial_dir, _ = execute("initial-validation", "validate")
            summary["initial_validation"] = metric(json.loads((initial_dir / "validation.json").read_text())[-1])
        fit_dir, record = execute(mode, "fit", checkpoint_input)
        observations = json.loads((fit_dir / "validation.json").read_text())
        checkpoints = json.loads((fit_dir / "checkpoints.json").read_text())
        selected = next(
            value
            for value in observations
            if value["epoch"] == checkpoints["metadata"]["best"]["epoch"]
            and value["global_step"] == checkpoints["metadata"]["best"]["global_step"]
        )
        if selected != min(observations, key=lambda value: (value["loss"], value["epoch"])):
            raise ValueError("Selected checkpoint does not match the lowest complete float64 validation CE / earliest tie")
        summary["metrics"]["val"] = metric(selected)
        immutable = output / "immutable"
        immutable.mkdir()
        for name, original in (("best", checkpoints["selected"]), ("last", checkpoints["last"])):
            path = immutable / f"{name}.ckpt"
            shutil.copyfile(original, path)
            path.chmod(0o444)
            metadata = {
                "path": str(path),
                "sha256": digest(path),
                **checkpoints["metadata"][name],
                "identities": controls,
                "parent_attempt_id": record["attempt_id"],
                "metrics": metric(selected if name == "best" else observations[-1]),
            }
            write_json(path.with_suffix(".json"), metadata)
            summary["checkpoints"][name] = metadata
        if mode == "continue":
            summary["parent_checkpoint"] = {
                "path": str(checkpoint_input),
                "sha256": parent_metadata["sha256"],
                "parent_attempt_id": options.parent_attempt_id,
            }
            summary["identities"]["parent_checkpoint"] = summary["parent_checkpoint"]
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "workflow.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
