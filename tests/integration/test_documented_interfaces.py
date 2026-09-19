"""Execute current guide snippets and test their explicitly documented contracts."""

import csv
import re
from pathlib import Path

import pytest
import torch
import yaml
from pytorch_lightning import LightningModule, Trainer
from sparkwheel import Config
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric

from lighter.callbacks import CsvWriter, FileWriter

GUIDES = Path(__file__).resolve().parents[2] / "docs/guides"


def example(path, language, marker):
    matches = [
        text for text in re.findall(rf"```{re.escape(language)}\n(.*?)```", path.read_text(), re.DOTALL) if marker in text
    ]
    assert len(matches) == 1, f"{path}: expected exactly one {language} example containing {marker!r}; found {len(matches)}"
    return matches[0]


def section(path, heading):
    matches = re.findall(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", path.read_text(), re.MULTILINE | re.DOTALL)
    assert len(matches) == 1, f"{path}: expected exactly one section {heading!r}; found {len(matches)}"
    return matches[0]


@pytest.mark.parametrize("count", [0, 2])
def test_documentation_extraction_rejects_missing_or_duplicate_examples(tmp_path, count):
    path = tmp_path / "guide.md"
    path.write_text("## Contract\n\n```yaml\nvalue: 3\n```\n" * count)
    with pytest.raises(AssertionError, match=rf"guide.md: expected exactly one yaml example.*found {count}"):
        example(path, "yaml", "value:")
    with pytest.raises(AssertionError, match=rf"guide.md: expected exactly one section.*found {count}"):
        section(path, "Contract")


@pytest.mark.parametrize(
    ("imports", "expression", "expected"),
    [
        ({"torch": "torch"}, "$torch.tensor(3).item()", 3),
        ({"Path": "pathlib.Path"}, "$str(Path('checkpoint.ckpt'))", "checkpoint.ckpt"),
    ],
)
def test_documented_import_declarations_resolve_real_expressions(imports, expression, expected):
    guidance = section(GUIDES / "configuration.md", "Functions, disabled components and imports")
    assert "`_imports_` supplies names to expressions." in guidance, "The documented expression-import rule is missing"
    # The guide states the rule in prose; these small probes and expected values are test-owned.
    actual = Config({"_imports_": imports, "probe": expression}).resolve("probe")
    assert type(actual) is type(expected)
    assert actual == expected


def test_documented_training_metrics_construct_independent_supported_states():
    source = example(GUIDES / "configuration.md", "yaml", "train_metrics:")
    config = Config(yaml.safe_load(source))
    train_metric = config.resolve("model::train_metrics")
    val_metric = config.resolve("model::val_metrics")
    assert isinstance(train_metric, Metric)
    assert isinstance(val_metric, Metric)
    assert train_metric is not val_metric
    train_metric.update(torch.tensor([1.0, 3.0]))
    val_metric.update(torch.tensor([10.0]))
    assert train_metric.compute().item() == pytest.approx(2)
    assert val_metric.compute().item() == pytest.approx(10)
    train_metric.reset()
    assert val_metric.compute().item() == pytest.approx(10)


def native_trainer(tmp_path, **kwargs):
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        **kwargs,
    )


def test_documented_writer_constructors_preserve_output_semantics(tmp_path, monkeypatch):
    callbacks = []
    for name in ("CsvWriter", "FileWriter"):
        source = example(GUIDES / "predictions.md", "yaml", f"lighter.callbacks.{name}")
        definitions = yaml.safe_load(source)["trainer"]["callbacks"]
        assert len(definitions) == 1, f"Expected one callback in the documented {name} example; found {len(definitions)}"
        callbacks.append(Config({"output_dir": str(tmp_path), "callback": definitions[0]}).resolve("callback"))
    csv_writer, file_writer = callbacks
    assert isinstance(csv_writer, CsvWriter)
    assert isinstance(file_writer, FileWriter)

    class PredictionTask(LightningModule):
        def predict_step(self, batch, batch_idx):
            return batch

    rows = [
        {"id": "0001", "prediction": torch.tensor(1.25), "target": torch.tensor(1.0)},
        {"id": "NA", "prediction": torch.tensor(-2.0), "target": torch.tensor(-1.0)},
    ]
    # Keep the FileWriter snippet's literal relative directory, rooted in an isolated working directory.
    monkeypatch.chdir(tmp_path)
    trainer = native_trainer(tmp_path, callbacks=callbacks)
    result = trainer.predict(PredictionTask(), DataLoader(rows, batch_size=2), return_predictions=False)
    assert result is None
    with (tmp_path / "predictions.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"id": "0001", "prediction": "1.25", "target": "1.0"},
            {"id": "NA", "prediction": "-2.0", "target": "-1.0"},
        ]
    directory = tmp_path / "outputs/tensors"
    assert {path.name for path in directory.iterdir()} == {"0001.pt", "NA.pt"}
    for name, expected in (("0001", 1.25), ("NA", -2.0)):
        assert torch.load(directory / f"{name}.pt", weights_only=True).item() == expected


def test_documented_native_optimizer_example_matches_analytic_sgd(tmp_path):
    source = example(GUIDES / "lightning-module.md", "python", "class RegressionTask(")
    namespace = {"__name__": __name__}
    exec(compile(source, str(GUIDES / "lightning-module.md"), "exec"), namespace)
    network = nn.Linear(2, 1)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[1.0, 2.0]]))
        network.bias.fill_(0.5)
    model = namespace["RegressionTask"](network, learning_rate=0.1)
    rows = [{"x": torch.tensor(x), "target": torch.tensor(0.0)} for x in ([1.0, 0.0], [0.0, 1.0])] * 2
    trainer = native_trainer(tmp_path)
    trainer.fit(model, DataLoader(rows, batch_size=2))
    assert model.automatic_optimization
    assert len(trainer.optimizers) == 1
    assert trainer.global_step == 2
    assert trainer.optimizers[0].param_groups[0]["lr"] == 0.1
    assert trainer.optimizers[0].param_groups[0]["momentum"] == 0.2
    # Mean-MSE gradients: ([1.5, 2.5], 4), then ([0.95, 1.85], 2.8).
    # The second SGD velocity includes 0.2 times the first gradient.
    torch.testing.assert_close(network.weight, torch.tensor([[0.725, 1.515]]))
    torch.testing.assert_close(network.bias, torch.tensor([-0.26]))
    assert trainer.callback_metrics["train/loss"].item() == pytest.approx(2.1625)


def test_documented_manual_multiple_optimizer_hooks_update_both_parameters(tmp_path):
    guidance = section(GUIDES / "lightning-module.md", "Hooks and custom optimization")
    entries = [line for line in guidance.splitlines() if line.startswith("| Manual or multiple-optimizer algorithm |")]
    assert len(entries) == 1, f"Expected one documented manual/multiple-optimizer hook entry; found {len(entries)}"
    for hook in ("automatic_optimization = False", "self.optimizers()", "self.manual_backward()"):
        assert f"`{hook}`" in entries[0], f"Missing documented manual-optimization hook: {hook}"

    # The guide names native hooks without prescribing an algorithm. This is a test-owned probe.
    class ManualTask(LightningModule):
        def __init__(self):
            super().__init__()
            self.first = nn.Parameter(torch.tensor(1.0))
            self.second = nn.Parameter(torch.tensor(2.0))
            self.automatic_optimization = False

        def configure_optimizers(self):
            return [torch.optim.SGD([self.first], lr=0.1), torch.optim.SGD([self.second], lr=0.2)]

        def training_step(self, batch, batch_idx):
            optimizers = self.optimizers()
            for name, parameter, optimizer, native in zip(
                ("first", "second"), (self.first, self.second), optimizers, self.trainer.optimizers, strict=True
            ):
                assert optimizer.optimizer is native
                optimizer.zero_grad()
                loss = parameter.square()
                self.manual_backward(loss)
                optimizer.step()
                self.log(f"train/{name}_loss", loss)

    model = ManualTask()
    trainer = native_trainer(tmp_path)
    trainer.fit(model, DataLoader(torch.ones(4), batch_size=2))
    assert not model.automatic_optimization
    assert len(trainer.optimizers) == 2
    assert trainer.global_step == 4
    # Each optimizer performs two independent theta <- theta - lr * 2 * theta updates.
    assert model.first.item() == pytest.approx(0.64)
    assert model.second.item() == pytest.approx(0.72)
    assert trainer.callback_metrics["train/first_loss"].item() == pytest.approx(0.64)
    assert trainer.callback_metrics["train/second_loss"].item() == pytest.approx(1.44)
