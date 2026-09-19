"""Native versus Lighter odd-population distributed prediction controls."""

import csv
import json
import os
import sys
from pathlib import Path

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lighter import LighterModule, Runner

IDS = ["00001", "NA", "λ,3", 'case"4', "line\n5"]


class Samples(Dataset):
    def __len__(self):
        return len(IDS)

    def __getitem__(self, index):
        return {"id": IDS[index], "value": torch.tensor(index, dtype=torch.float64)}


class PredictionSteps:
    def predict_step(self, batch, batch_idx):
        return {"id": batch["id"], "prediction": batch["value"] * 3 + 1}

    def predict_dataloader(self):
        dataset = Samples()
        sampler = (
            DistributedSampler(
                dataset, num_replicas=self.trainer.world_size, rank=self.global_rank, shuffle=False, drop_last=False
            )
            if self.policy == "explicit-padding"
            else None
        )
        return DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)


class NativeTask(PredictionSteps, LightningModule):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy


class LighterTask(PredictionSteps, LighterModule):
    def __init__(self, policy):
        super().__init__(network=torch.nn.Identity())
        self.policy = policy


class Audit(Callback):
    def __init__(self, out):
        self.out = out
        self.rows = []

    def on_predict_start(self, trainer, module):
        loader = trainer.predict_dataloaders
        effective_sampler = loader.batch_sampler.sampler
        self.sampler = f"{type(effective_sampler).__module__}.{type(effective_sampler).__qualname__}"
        self.indices = list(effective_sampler)

    def on_predict_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        self.rows.extend(
            {"id": name, "prediction": float(value)}
            for name, value in zip(outputs["id"], outputs["prediction"].cpu().tolist(), strict=True)
        )

    def on_predict_end(self, trainer, module):
        Path(self.out, f"rank-{trainer.global_rank}.json").write_text(
            json.dumps(
                {
                    "rank": trainer.global_rank,
                    "pid": os.getpid(),
                    "sampler": self.sampler,
                    "indices": self.indices,
                    "rows": self.rows,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    kind, policy = sys.argv[1:3]
    out = Path(sys.argv[3]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    options = dict(
        accelerator="cpu",
        devices=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(out),
    )
    if kind == "native":
        trainer = Trainer(
            **options, strategy=DDPStrategy(start_method="spawn", process_group_backend="gloo"), callbacks=[Audit(str(out))]
        )
        result = trainer.predict(NativeTask(policy), return_predictions=False)
    elif kind == "lighter":
        source = {
            "seed": 19,
            "model": {"_target_": "__main__.LighterTask", "policy": policy},
            "trainer": {
                "_target_": "pytorch_lightning.Trainer",
                **options,
                "strategy": {
                    "_target_": "pytorch_lightning.strategies.DDPStrategy",
                    "start_method": "spawn",
                    "process_group_backend": "gloo",
                },
                "callbacks": [
                    {"_target_": "__main__.Audit", "out": str(out)},
                    {
                        "_target_": "lighter.callbacks.CsvWriter",
                        "path": str(out / "predictions.csv"),
                        "keys": ["id", "prediction"],
                    },
                ],
            },
        }
        runner = Runner()
        result = runner.run("predict", [source], return_predictions=False)
        assert runner.last_run_path.is_file()
    else:
        raise ValueError(kind)
    assert result is None
    ranks = [json.loads((out / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["pid"] != ranks[1]["pid"]
    rows = [row for rank in ranks for row in rank["rows"]]
    expected_indices = [[0, 2, 4], [1, 3, 0] if policy == "explicit-padding" else [1, 3]]
    assert [rank["indices"] for rank in ranks] == expected_indices
    expected = [{"id": IDS[index], "prediction": float(index * 3 + 1)} for indices in expected_indices for index in indices]
    assert rows == expected
    if kind == "lighter":
        with (out / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert [{"id": row["id"], "prediction": float(row["prediction"])} for row in csv_rows] == expected
        assert not list(out.glob("*.tmp_rank*.csv"))
        record = json.loads(runner.last_run_path.read_text())
        assert record["status"] == "completed" and record["execution"]["world_size"] == 2
        assert record["prediction_destinations"][0]["exists_at_end"] is True
    summary = {
        "passed": True,
        "kind": kind,
        "policy": policy,
        "rows": rows,
        "sampler": ranks[0]["sampler"],
        "rank_indices": expected_indices,
        "native_return": None,
        "source_population_size": len(IDS),
        "emitted_rows": len(rows),
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
