"""Execute the specific public documentation interfaces repaired by the audit."""

import re
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from sparkwheel import Config
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Metric, MetricCollection

GUIDES = Path(__file__).resolve().parents[2] / "docs/guides"


def blocks(path, language):
    return re.findall(rf"```{language}\n(.*?)```", path.read_text(), re.DOTALL)


def test_documented_import_declarations_resolve_real_expressions():
    checked = 0
    for text in blocks(GUIDES / "configuration.md", "yaml"):
        if "_imports_:" not in text:
            continue
        document = yaml.safe_load(text)
        if not isinstance(document, dict) or "_imports_" not in document:
            continue
        imports = document["_imports_"]
        assert isinstance(imports, dict)
        expression = "$torch.tensor(3).item()" if "torch" in imports else "$str(Path('checkpoint.ckpt'))"
        expected = 3 if "torch" in imports else "checkpoint.ckpt"
        assert Config({"_imports_": imports, "probe": expression}).resolve("probe") == expected
        checked += 1
    assert checked == 3


def test_documented_training_metrics_construct_supported_types():
    checked = 0
    for text in blocks(GUIDES / "configuration.md", "yaml"):
        if "train_metrics:" not in text:
            continue
        document = yaml.safe_load(text)
        if not isinstance(document, dict):
            continue
        model = document.get("model", {})
        if not isinstance(model, dict) or "train_metrics" not in model:
            continue
        config = Config({"vars": {"num_classes": 10}, "metric": model["train_metrics"]})
        assert isinstance(config.resolve("metric"), (Metric, MetricCollection))
        checked += 1
    assert checked == 3


def test_documented_writer_constructors_match_public_api():
    checked = 0
    for path in (GUIDES / "training.md", GUIDES.parent / "reference/cli.md", GUIDES.parent / "faq.md"):
        for text in blocks(path, "yaml"):
            if "lighter.callbacks." not in text:
                continue
            document = yaml.safe_load(text)
            if not isinstance(document, dict):
                continue
            for callback in document.get("trainer", {}).get("callbacks", []):
                if callback.get("_target_", "") in {"lighter.callbacks.CsvWriter", "lighter.callbacks.FileWriter"}:
                    Config({"callback": callback}).resolve("callback")
                    checked += 1
    assert checked == 5


def test_documented_manual_multiple_optimizer_example_runs_native_trainer(tmp_path):
    source = next(text for text in blocks(GUIDES / "lightning-module.md", "python") if "class GAN(" in text)
    namespace = {"__name__": __name__}
    exec(compile(source, str(GUIDES / "lightning-module.md"), "exec"), namespace)
    generator = nn.Linear(2, 2)
    generator.latent_dim = 2
    discriminator = nn.Linear(2, 1)
    model = namespace["GAN"](generator, discriminator)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4)), batch_size=2))
    assert not model.automatic_optimization
    assert len(trainer.optimizers) == 2
    assert trainer.global_step == 4
    assert {"train/g_loss", "train/d_loss"} <= trainer.callback_metrics.keys()
    assert all(not torch.equal(before, after) for before, after in zip(initial, model.parameters(), strict=True))
