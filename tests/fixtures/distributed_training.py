"""Two-rank managed optimizer setup against mathematical and native SGD controls."""

import json
import math
import sys
from pathlib import Path

import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, DistributedSampler

from lighter import LighterModule, Runner


class Network(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, dtype=torch.float64)
        with torch.no_grad():
            self.weight.fill_(1.0)
            self.bias.fill_(0.5)


class Task(LighterModule):
    def training_step(self, batch, batch_idx):
        return self.network(batch.reshape(-1, 1)).square().mean()

    def train_dataloader(self):
        data = torch.arange(1, 9, dtype=torch.float64)
        sampler = DistributedSampler(data, num_replicas=self.trainer.world_size, rank=self.global_rank, shuffle=False)
        return DataLoader(data, batch_size=1, sampler=sampler)


class Audit(Callback):
    def __init__(self, model, out):
        self.model, self.out, self.records = model, out, []
        self.initially_unbuilt = model.optimizer is None

    def on_fit_start(self, trainer, model):
        assert self.initially_unbuilt
        assert model is self.model
        assert model._optimizer_binding is not None
        assert model.optimizer is trainer.optimizers[0]
        assert model.optimizer.param_groups[0]["params"][0] is model.network.weight
        assert model.optimizer.param_groups[1]["params"][0] is model.network.bias
        self.initial = [float(model.network.weight.detach()), float(model.network.bias.detach())]

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.records.append(
            {"x": float(batch.item()), "weights": [float(model.network.weight.detach()), float(model.network.bias.detach())]}
        )

    def on_train_end(self, trainer, model):
        recorder = next(c for c in trainer.callbacks if type(c).__name__ == "RunRecorder")
        Path(self.out, f"rank-{trainer.global_rank}.json").write_text(
            json.dumps({"initial": self.initial, "records": self.records, "record_path": str(recorder.path)}, indent=2)
        )


if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = {
        "seed": 19,
        "model": {
            "_target_": "__main__.Task",
            "network": {"_target_": "__main__.Network"},
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "params": [
                    {"params": "$[@model.network.weight]", "lr": 0.01},
                    {"params": "$[@model::network.bias]", "lr": 0.02},
                ],
                "lr": 0.1,
                "momentum": 0.2,
                "weight_decay": 0.1,
            },
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "accelerator": "cpu",
            "devices": 2,
            "max_steps": 4,
            "strategy": {
                "_target_": "pytorch_lightning.strategies.DDPStrategy",
                "start_method": "spawn",
                "process_group_backend": "gloo",
            },
            "callbacks": [{"_target_": "__main__.Audit", "model": "@model", "out": str(out)}],
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "default_root_dir": str(out),
        },
    }
    runner = Runner()
    assert runner.run("fit", [source]) is None
    records = list(out.glob("lighter_runs/*/record.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "completed" and record["execution"]["world_size"] == 2
    assert record["observed_start"]["global_step"] == 0 and record["observed_end"]["global_step"] == 4
    assert [g["settings"]["lr"] for g in record["observed_start"]["optimizers"][0]["groups"]] == [0.01, 0.02]
    (out / "parent-record-path.json").write_text(
        json.dumps(
            {
                "last_run_path": str(runner.last_run_path) if runner.last_run_path else None,
                "published_record": str(records[0]),
            },
            indent=2,
        )
    )
    assert runner.last_run_path == records[0]
    ranks = [json.loads((out / f"rank-{i}.json").read_text()) for i in range(2)]
    assert ranks[0]["initial"] == ranks[1]["initial"] == [1.0, 0.5]
    assert ranks[0]["record_path"] == ranks[1]["record_path"] == str(records[0])
    assert [r["x"] for r in ranks[0]["records"]] == [1, 3, 5, 7]
    assert [r["x"] for r in ranks[1]["records"]] == [2, 4, 6, 8]
    expected = []
    w, b = 1.0, 0.5
    mw, mb = 0.0, 0.0
    native = Network()
    optimizer = torch.optim.SGD(
        [{"params": [native.weight], "lr": 0.01}, {"params": [native.bias], "lr": 0.02}],
        lr=0.1,
        momentum=0.2,
        weight_decay=0.1,
    )
    for i in range(4):
        xs = [rank["records"][i]["x"] for rank in ranks]
        ys = [w * x + b for x in xs]
        mw = 0.2 * mw + sum(2 * y * x for x, y in zip(xs, ys, strict=True)) / 2 + 0.1 * w
        mb = 0.2 * mb + sum(2 * y for y in ys) / 2 + 0.1 * b
        w -= 0.01 * mw
        b -= 0.02 * mb
        expected.append([w, b])
        optimizer.zero_grad()
        native(torch.tensor(xs, dtype=torch.float64).reshape(-1, 1)).square().mean().backward()
        optimizer.step()
        reference = [float(native.weight.detach()), float(native.bias.detach())]
        assert all(math.isclose(a, z, rel_tol=1e-12, abs_tol=1e-12) for a, z in zip(reference, expected[-1], strict=True))
        for rank in ranks:
            actual = rank["records"][i]["weights"]
            assert all(math.isclose(a, z, rel_tol=1e-12, abs_tol=1e-12) for a, z in zip(actual, expected[-1], strict=True)), (
                actual,
                expected[-1],
            )
    assert [r["weights"] for r in ranks[0]["records"]] == [r["weights"] for r in ranks[1]["records"]]
    result = {
        "passed": True,
        "world_size": 2,
        "backend": "gloo",
        "parameters_bound_at_native_setup": True,
        "custom_group_learning_rates": [0.01, 0.02],
        "expected_weights_by_step": expected,
        "native_torch_optimizer_and_analytic_controls_agree": True,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
