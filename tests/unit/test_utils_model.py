import torch
from torch.nn import Linear, Sequential

from lighter.utils.model import remove_n_last_layers_sequentially, replace_layer_with, replace_layer_with_identity


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(10, 10)
        self.layer2 = Linear(10, 10)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def test_replace_layer_with():
    model = DummyModel()
    new_layer = Linear(10, 4)
    replace_layer_with(model, "layer1", new_layer)
    assert model.layer1 == new_layer


def test_replace_layer_with_identity():
    model = DummyModel()
    replace_layer_with_identity(model, "layer1")
    assert isinstance(model.layer1, torch.nn.Identity)


def test_remove_n_last_layers_sequentially():
    model = Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))
    new_model = remove_n_last_layers_sequentially(model, num_layers=1)
    assert len(new_model) == 2

    model = Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))
    new_model = remove_n_last_layers_sequentially(model, num_layers=2)
    assert len(new_model) == 1
