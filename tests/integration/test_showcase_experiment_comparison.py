"""Small structural and adversarial checks for the comparison's public contract."""

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


@pytest.fixture
def example(monkeypatch):
    project_root = Path(__file__).resolve().parents[2] / "projects"
    monkeypatch.syspath_prepend(str(project_root))
    return SimpleNamespace(
        **{
            name: importlib.import_module(f"experiment_comparison.{name}")
            for name in ("task", "data", "workflow", "artifacts")
        }
    )


@pytest.fixture
def fixture_data(tmp_path, example):
    populations = {}
    for population, count in (("train", 5), ("val", 5), ("test", 5)):
        rows = [
            {"id": value, "label": index % 3, "original_index": index, "source_split": population}
            for index, value in enumerate(("00001", "NA", "NULL", "case,4", "λ-5"))
        ]
        manifest = tmp_path / f"{population}.json"
        example.artifacts.write_json(
            manifest, {"schema_version": 1, "population": population, "source_split": population, "rows": rows}
        )
        images = np.zeros((count, 3, 32, 32), dtype=np.uint8)
        images[:, 0] = 255
        images[:, 1] = 128
        image_path = tmp_path / f"{population}.npy"
        np.save(image_path, images, allow_pickle=False)
        populations[population] = {
            "manifest": manifest.name,
            "manifest_sha256": example.artifacts.sha256(manifest),
            "images": image_path.name,
            "images_sha256": example.artifacts.sha256(image_path),
            "count": count,
        }
    path = tmp_path / "data.json"
    example.artifacts.write_json(
        path, {"schema_version": 1, "dataset": "synthetic-contract-fixture", "populations": populations}
    )
    return path


def test_known_population_tail_and_sensitive_wrong_mean(example):
    logits = torch.zeros(5, 10)
    logits[torch.arange(5), torch.tensor([0, 1, 0, 2, 1])] = math.log(4)
    labels = torch.tensor([0, 1, 1, 2, 2])
    totals = example.task.EvaluationTotals()
    for start in (0, 2, 4):
        totals.update(logits[start : start + 2], labels[start : start + 2])
    result = totals.result()
    expected = math.log(13) - 0.6 * math.log(4)
    assert result["count"] == 5 and result["accuracy"] == 0.6
    assert result["loss"] == pytest.approx(expected, abs=2e-7)
    expected_confusion = [[0] * 10 for _ in range(10)]
    expected_confusion[0][0] = expected_confusion[1][1] = expected_confusion[1][0] = 1
    expected_confusion[2][2] = expected_confusion[2][1] = 1
    assert result["confusion"] == expected_confusion
    wrong = torch.stack([torch.nn.functional.cross_entropy(logits[s : s + 2], labels[s : s + 2]) for s in (0, 2, 4)]).mean()
    assert abs(wrong.item() - expected) > 0.1
    dropped = example.task.EvaluationTotals()
    dropped.update(logits[:4], labels[:4])
    assert dropped.result()["count"] != result["count"]


def test_data_bytes_identity_tail_and_sealed_setup(example, fixture_data):
    data = example.data.ComparisonData(fixture_data, seed=17, batch_size=2)
    data.setup("fit")
    assert set(data.populations) == {"train", "val"}
    batches = list(data.val_dataloader())
    assert [len(batch["id"]) for batch in batches] == [2, 2, 1]
    assert [value for batch in batches for value in batch["id"]] == ["00001", "NA", "NULL", "case,4", "λ-5"]
    image = batches[0]["image"][0]
    assert image.dtype == torch.float32 and image.shape == (3, 32, 32)
    assert torch.all(image[0] == 1) and torch.all(image[2] == 0)
    assert torch.all(image[1] == torch.tensor(128 / 255, dtype=torch.float32))


@pytest.mark.parametrize("fault", ["duplicate", "numeric", "hash", "layout"])
def test_invalid_data_is_rejected(example, fixture_data, fault):
    contents = json.loads(fixture_data.read_text())
    entry = contents["populations"]["val"]
    path = fixture_data.parent / entry["manifest"]
    manifest = json.loads(path.read_text())
    if fault == "duplicate":
        manifest["rows"][1]["id"] = manifest["rows"][0]["id"]
    elif fault == "numeric":
        manifest["rows"][0]["id"] = 1
    elif fault == "hash":
        manifest["rows"][0]["label"] = 8
    else:
        image_path = fixture_data.parent / entry["images"]
        np.save(image_path, np.zeros((5, 32, 32, 3), dtype=np.uint8))
        entry["images_sha256"] = example.artifacts.sha256(image_path)
    example.artifacts.write_json(path, manifest)
    if fault != "hash":
        entry["manifest_sha256"] = example.artifacts.sha256(path)
    example.artifacts.write_json(fixture_data, contents)
    with pytest.raises(ValueError):
        example.data.ImagePopulation(fixture_data, "val")


def test_order_is_independent_of_model_rng_and_rejects_missing_tail(example, fixture_data, tmp_path):
    dataset = example.data.ImagePopulation(fixture_data, "train")
    order = example.data.EpochOrder(dataset.rows, 17, 2)
    before = list(order)
    torch.rand(37)
    assert list(order) == before
    order.set_epoch(1)
    generator = torch.Generator().manual_seed(17 * 100000 + 2)
    assert list(order) == torch.randperm(5, generator=generator).tolist()
    path = tmp_path / "order.json"
    ids = [row["id"] for row in dataset.rows]
    payload = {
        "seed": 17,
        "batch_size": 2,
        "data_manifest_sha256": example.artifacts.sha256(fixture_data),
        "epochs": [{"epoch": 1, "batches": [ids[:2], ids[2:4], ids[4:]]}],
    }
    example.artifacts.write_json(path, payload)
    sampler = example.data.EpochOrder(dataset.rows, 17, 2, path, fixture_data)
    assert [len(batch["id"]) for batch in DataLoader(dataset, sampler=sampler, batch_size=2)] == [2, 2, 1]
    payload["epochs"][0]["batches"].pop()
    example.artifacts.write_json(path, payload)
    with pytest.raises(ValueError, match="retained tail"):
        example.data.EpochOrder(dataset.rows, 17, 2, path, fixture_data)


def test_native_checkpoint_tie_retains_earliest():
    callback = ModelCheckpoint(monitor="val/ce", mode="min", save_top_k=1)
    callback.best_k_models = {"epoch-1.ckpt": torch.tensor(2.0)}
    callback.kth_best_model_path = "epoch-1.ckpt"
    trainer = SimpleNamespace(strategy=SimpleNamespace(reduce_boolean_decision=bool))
    assert not callback.check_monitor_top_k(trainer, torch.tensor(2.0))
    assert callback.check_monitor_top_k(trainer, torch.tensor(1.9))


def test_checkpoint_digest_and_mode_precedence(example, tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"immutable selected checkpoint")
    example.artifacts.write_json(
        checkpoint.with_suffix(".json"), {"sha256": example.artifacts.sha256(checkpoint), "identities": {}}
    )
    assert example.workflow.check_checkpoint(checkpoint, {})["sha256"] == example.artifacts.sha256(checkpoint)
    checkpoint.write_bytes(b"deliberately wrong checkpoint")
    with pytest.raises(ValueError, match="Checkpoint digest"):
        example.workflow.check_checkpoint(checkpoint, {})
    with pytest.raises(SystemExit):
        example.workflow.build_parser().parse_args(
            ["--output-dir", str(tmp_path / "out"), "--evaluate-from", "a", "--continue-from", "b"]
        )
    options = example.workflow.build_parser().parse_args(["--output-dir", str(tmp_path / "out"), "--evaluation", "final"])
    with pytest.raises(ValueError, match="explicit selected checkpoint"):
        example.workflow.run(options)
    assert "torch" not in example.workflow.__dict__


def test_network_shape_and_restore_initialization_precedence(example, tmp_path):
    network = example.task.Classifier()
    assert set(network.state_dict()) == {"0.weight", "0.bias", "3.weight", "3.bias", "7.weight", "7.bias"}
    different = {key: torch.full_like(value, 0.123) for key, value in network.state_dict().items()}
    initial = tmp_path / "initial.pt"
    torch.save(different, initial)
    restored = example.task.Classifier(initial)
    restored.load_state_dict(network.state_dict())  # Native checkpoint restore order.
    assert all(torch.equal(restored.state_dict()[key], value) for key, value in network.state_dict().items())
    assert restored(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_near_tie_selection_keeps_float64_precision(example):
    first = 2.0
    second = 2.0 - 1e-9
    assert torch.tensor(first, dtype=torch.float32) == torch.tensor(second, dtype=torch.float32)
    callback = ModelCheckpoint(monitor="val/ce", mode="min", save_top_k=1)
    callback.best_k_models = {"first.ckpt": torch.tensor(first, dtype=torch.float64)}
    callback.kth_best_model_path = "first.ckpt"
    trainer = SimpleNamespace(strategy=SimpleNamespace(reduce_boolean_decision=bool))
    assert callback.check_monitor_top_k(trainer, torch.tensor(second, dtype=torch.float64))
    observed = {}
    module = SimpleNamespace(
        validation_totals=SimpleNamespace(result=lambda: {"loss": second, "accuracy": 0.6}),
        device=torch.device("cpu"),
        log=lambda key, value, **kwargs: observed.update({key: value}),
    )
    example.task.ScientificSteps.on_validation_epoch_end(module)
    assert observed["val/ce"].dtype == torch.float64 and observed["val/ce"].item() == second


def test_control_digest_is_checked_before_scientific_launch(example, fixture_data, tmp_path):
    initial = tmp_path / "initial_state.pt"
    order = tmp_path / "batch_order.json"
    initial.write_bytes(b"declared initialization")
    example.artifacts.write_json(order, {"batch_size": 2, "data_manifest_sha256": example.artifacts.sha256(fixture_data)})
    controls = {
        "seed": 17,
        "case": {"revision": 1},
        "data_manifest": {"sha256": example.artifacts.sha256(fixture_data)},
        "initial_state": {"sha256": example.artifacts.sha256(initial)},
        "batch_order": {"sha256": example.artifacts.sha256(order)},
    }
    example.artifacts.write_json(tmp_path / "controls.json", controls)
    options = SimpleNamespace(data_manifest=fixture_data, initial_state=initial, order_manifest=order, seed=17, batch_size=2)
    assert example.workflow.check_controls(options)["initial_state"]["sha256"] == controls["initial_state"]["sha256"]
    initial.write_bytes(b"different initialization")
    with pytest.raises(ValueError, match="Control artifact digest mismatch"):
        example.workflow.check_controls(options)


def test_scientific_output_directory_cannot_be_reused(example, tmp_path):
    callback = example.artifacts.ExperimentArtifacts(tmp_path)
    callback.setup(None, None, "fit")
    previous = (tmp_path / "runtime.json").read_bytes()
    with pytest.raises(FileExistsError):
        callback.setup(None, None, "test")
    callback.teardown(None, None, "test")
    assert (tmp_path / "runtime.json").read_bytes() == previous


def test_learning_rate_choice_composes_without_managed_optimizer_dependency(example):
    from lighter.engine.runner import ConfigLoader

    project = Path(example.task.__file__).parent
    config = ConfigLoader.load([str(project / "config.yaml"), str(project / "high_lr.yaml")])
    assert config.resolve("learning_rate") == 0.1
    assert config.resolve("trainer::callbacks::1::requested_lr") == 0.1


@pytest.mark.parametrize("parent", [None, "parent-attempt-123"])
def test_ordinary_and_continued_run_options_satisfy_record_contract(example, tmp_path, parent):
    from lighter.engine.records import RunRecorder
    from lighter.engine.runner import ConfigLoader

    inputs = [str(Path(example.task.__file__).parent / "config.yaml"), f"output_dir={tmp_path}"]
    if parent is not None:
        inputs.append(f"run::parent_attempt_id={parent}")
    config = ConfigLoader.load(inputs)
    recorder = RunRecorder(
        source=config.get(), stage="fit", seed=17, requested_args={}, inputs=inputs, options=config.resolve("run")
    )
    assert recorder.options.get("parent_attempt_id") == parent


@pytest.mark.parametrize("backend", ["lighter", "native"])
@pytest.mark.parametrize("accelerator,precision", [("cpu", "32-true"), ("cuda", "16-mixed")])
def test_workflow_forwards_device_and_precision_to_scientific_child(
    example, fixture_data, tmp_path, monkeypatch, backend, accelerator, precision
):
    class ScientificLaunch(Exception):
        pass

    observed = []

    def intercept(command, **kwargs):
        if "inspect" in command:
            return SimpleNamespace(stdout="{}", returncode=0)
        observed.append(command)
        raise ScientificLaunch

    monkeypatch.setattr(example.workflow.subprocess, "run", intercept)
    options = example.workflow.build_parser().parse_args(
        [
            "--backend",
            backend,
            "--data-manifest",
            str(fixture_data),
            "--output-dir",
            str(tmp_path / "workflow"),
            "--accelerator",
            accelerator,
            "--precision",
            precision,
        ]
    )
    with pytest.raises(ScientificLaunch):
        example.workflow.run(options)
    (command,) = observed
    if backend == "lighter":
        assert f'trainer::accelerator="{accelerator}"' in command
        assert f'trainer::precision="{precision}"' in command
    else:
        assert command[command.index("--accelerator") + 1] == accelerator
        assert command[command.index("--precision") + 1] == precision


def test_native_runner_passes_explicit_precision_and_preserves_defaults(example, fixture_data, tmp_path, monkeypatch):
    native = importlib.import_module("experiment_comparison.native")
    observed = []

    class TrainerConstructed(Exception):
        pass

    def trainer(**kwargs):
        observed.append(kwargs)
        raise TrainerConstructed

    monkeypatch.setattr(native.pl, "Trainer", trainer)
    for name, arguments in (("default", []), ("cuda", ["--accelerator", "cuda", "--precision", "16-mixed"])):
        with pytest.raises(TrainerConstructed):
            native.main(["fit", "--data-manifest", str(fixture_data), "--output-dir", str(tmp_path / name), *arguments])
    assert [(item["accelerator"], item["precision"], item["devices"]) for item in observed] == [
        ("cpu", "32-true", 1),
        ("cuda", "16-mixed", 1),
    ]
    options = example.workflow.build_parser().parse_args(["--output-dir", str(tmp_path / "defaults")])
    assert (options.accelerator, options.precision) == ("cpu", "32-true")


@pytest.mark.parametrize("backend", ["lighter", "native"])
def test_workflow_rejects_a_successful_child_with_precision_fallback(example, fixture_data, tmp_path, monkeypatch, backend):
    output = tmp_path / "workflow"

    def child(command, **kwargs):
        if "inspect" in command:
            return SimpleNamespace(stdout="{}", returncode=0)
        stage = output / "initial-validation"
        record = stage / "lighter_runs" / "attempt" / "record.json" if backend == "lighter" else stage / "native-attempt.json"
        record.parent.mkdir(parents=True)
        example.artifacts.write_json(record, {"attempt_id": "attempt", "status": "completed"})
        example.artifacts.write_json(stage / "runtime.json", {"execution": {"device": "cpu", "precision": "bf16-mixed"}})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(example.workflow.subprocess, "run", child)
    options = example.workflow.build_parser().parse_args(
        ["--backend", backend, "--data-manifest", str(fixture_data), "--output-dir", str(output), "--precision", "16-mixed"]
    )
    with pytest.raises(ValueError, match="different device or precision"):
        example.workflow.run(options)
    receipt = json.loads((output / "workflow.json").read_text())
    assert receipt["stages"][0]["exit_code"] == 0  # Process success cannot certify the requested precision.
    assert receipt["requested_precision"] == "16-mixed"
