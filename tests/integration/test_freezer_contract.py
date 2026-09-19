"""Freezing owns selected parameters and survives native checkpoint continuation."""

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from lighter.callbacks import Freezer


class FreezeTask(LightningModule):
    def __init__(self, teacher_frozen=True):
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.tensor(1.0))
        self.head = torch.nn.Parameter(torch.tensor(1.0))
        self.teacher = torch.nn.Parameter(torch.tensor(1.0), requires_grad=not teacher_frozen)
        self.observed = []

    def training_step(self, batch, batch_idx):
        self.observed.append((self.global_step, self.backbone.requires_grad, self.teacher.requires_grad))
        return (self.backbone + self.head + self.teacher).square()

    def train_dataloader(self):
        return DataLoader([0, 0, 0, 0], batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)


def trainer(tmp_path, callbacks, **kwargs):
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=callbacks,
        default_root_dir=str(tmp_path),
        num_sanity_val_steps=0,
        **kwargs,
    )


def test_freezer_preserves_unselected_and_excepted_original_flags():
    task = FreezeTask()
    callback = Freezer(names=["backbone", "teacher"], except_names=["teacher"])
    task.backbone.grad = torch.ones_like(task.backbone)
    callback._set_model_requires_grad(task, False)
    assert not task.backbone.requires_grad
    assert task.backbone.grad is None
    assert not task.teacher.requires_grad
    callback._set_model_requires_grad(task, True)
    assert task.backbone.requires_grad
    assert not task.teacher.requires_grad


def test_release_preserves_selected_prefrozen_parameter():
    task = FreezeTask()
    callback = Freezer(names=["backbone", "teacher"])
    callback._set_model_requires_grad(task, False)
    callback._set_model_requires_grad(task, True)
    assert task.backbone.requires_grad
    assert not task.teacher.requires_grad


def test_no_match_is_an_error_instead_of_silent_success(tmp_path):
    with pytest.raises(ValueError, match="matched no parameters"):
        trainer(tmp_path, [Freezer(names="backbon")], max_steps=1).fit(FreezeTask())


def test_overlapping_ownership_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        trainer(tmp_path, [Freezer(names="backbone"), Freezer(name_starts_with="back")], max_steps=1).fit(FreezeTask())


def test_disjoint_freezers_and_tied_parameter_aliases(tmp_path):
    task = FreezeTask()
    task.alias = task.backbone
    first = Freezer(names="alias", until_step=1)
    second = Freezer(names="teacher")
    trainer(tmp_path, [first, second], max_steps=2).fit(task)
    assert task.observed == [(0, False, False), (1, True, False)]


class SaveAndStop(Callback):
    def __init__(self, path):
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.path)


@pytest.mark.parametrize("resume_steps", [2, 4])
def test_native_checkpoint_continuation_restores_freeze_schedule(tmp_path, resume_steps):
    path = tmp_path / "freeze.ckpt"
    first = FreezeTask()
    trainer(tmp_path, [Freezer(names="backbone", until_step=3), SaveAndStop(path)], max_epochs=1, limit_train_batches=1).fit(
        first
    )
    assert first.backbone.item() == 1.0
    resumed = FreezeTask(teacher_frozen=True)
    engine = trainer(tmp_path, [Freezer(names="backbone", until_step=3)], max_steps=resume_steps, limit_train_batches=1)
    engine.fit(resumed, ckpt_path=str(path))
    assert resumed.teacher.item() == 1.0
    assert all(frozen is (step >= 3) for step, frozen, _ in resumed.observed)
    assert all(not teacher for _, _, teacher in resumed.observed)
    if resume_steps <= 3:
        assert resumed.backbone.item() == 1.0
    else:
        assert resumed.backbone.item() != 1.0


def test_reactivated_parameters_must_already_belong_to_optimizer(tmp_path):
    class Filtered(FreezeTask):
        def configure_optimizers(self):
            return torch.optim.SGD([self.head], lr=0.01)

    with pytest.raises(ValueError, match="optimizer"):
        trainer(tmp_path, [Freezer(names="backbone", until_step=1)], max_steps=2).fit(Filtered())
