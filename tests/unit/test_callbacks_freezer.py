import logging
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from lighter.callbacks.freezer import Freezer
from lighter.model import LighterModule


class Task(LighterModule):
    def training_step(self, batch, batch_idx):
        return self(batch[0]).sum()


def network():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 1))


def task():
    return Task(network=network(), optimizer=lambda parameters: torch.optim.SGD(parameters, lr=0.1))


def at(step=0, epoch=0):
    """The trainer state the Freezer reads."""
    return SimpleNamespace(global_step=step, current_epoch=epoch)


def requires_grad(model):
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def test_freezer_validates_its_arguments():
    with pytest.raises(ValueError, match="At least one of `names` or `name_starts_with` must be specified."):
        Freezer()
    with pytest.raises(ValueError, match="Only one of `until_step` or `until_epoch` can be specified."):
        Freezer(names="0.weight", until_step=10, until_epoch=1)
    assert Freezer(names="0.weight").names == ["0.weight"]


def test_freezer_freezes_selected_parameters_during_fit():
    model = task()
    loader = DataLoader(TensorDataset(torch.randn(8, 4)), batch_size=4)
    trainer = Trainer(callbacks=[Freezer(name_starts_with="0.")], max_epochs=1, logger=False, enable_checkpointing=False)
    initial = model.network[0].weight.clone()
    trainer.fit(model, loader)
    assert torch.equal(model.network[0].weight, initial)
    assert requires_grad(model.network) == {
        "0.weight": False,
        "0.bias": False,
        **{f"{i}.{p}": True for i in (1, 2) for p in ("weight", "bias")},
    }


@pytest.mark.parametrize("until, text", [({"until_step": 2}, "until step 2"), ({"until_epoch": 1}, "until epoch 1")])
def test_freezer_unfreezes_when_the_step_or_epoch_is_reached(until, text, caplog):
    model, freezer = task(), Freezer(names=["0.weight", "1.weight"], **until)
    with caplog.at_level(logging.INFO, logger="lighter"):
        freezer.on_train_batch_start(at(), model, None, 0)
        assert not model.network[0].weight.requires_grad and not model.network[1].weight.requires_grad
        freezer.on_train_batch_start(at(step=1), model, None, 1)
        assert not model.network[0].weight.requires_grad
        freezer.on_train_batch_start(at(step=2, epoch=1), model, None, 2)
    assert all(requires_grad(model).values())
    assert caplog.messages == [f"Froze ['0.weight', '1.weight'] {text}.", "Unfroze ['0.weight', '1.weight']."]


def test_freezer_keeps_the_users_own_frozen_parameters():
    model, freezer = task(), Freezer(name_starts_with="0.", until_step=1)
    model.network[0].bias.requires_grad = False  # frozen by the user, e.g. in their model's __init__
    model.network[2].weight.requires_grad = False  # not selected: never touched
    freezer.on_train_batch_start(at(), model, None, 0)
    freezer.on_train_batch_start(at(step=1), model, None, 1)
    assert requires_grad(model.network) == {
        "0.weight": True, "0.bias": False, "1.weight": True, "1.bias": True, "2.weight": False, "2.bias": True
    }  # fmt: skip


def test_freezer_exceptions_are_left_alone():
    model = task()
    Freezer(name_starts_with=["0.", "1."], except_names="0.bias", except_name_starts_with="1.b").on_train_batch_start(
        at(), model, None, 0
    )
    assert [name for name, flag in requires_grad(model.network).items() if not flag] == ["0.weight", "1.weight"]


def test_freezer_works_on_plain_lightning_modules():
    model = task().network
    module = SimpleNamespace(named_parameters=model.named_parameters)
    Freezer(names="2.bias").on_train_batch_start(at(), module, None, 0)
    assert not model[2].bias.requires_grad


def test_freezer_that_matches_nothing_is_an_error():
    with pytest.raises(ValueError, match=r"Freezer matched no parameter. The model's parameter names are like \['0.weight'"):
        Freezer(names="network.0.weight").on_train_batch_start(at(), task(), None, 0)
