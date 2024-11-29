import pytest
from lighter.utils.model import replace_layer_with_identity
import torch
from torch.nn import Linear
from lighter.system import LighterSystem

class DummySystem(LighterSystem):
    def __init__(self):
        super().__init__()
        self.fc = Linear(10, 10)

def test_replace_layer_with_identity():
    model = DummyModel()
    replace_layer_with_identity(model, "fc")
    assert isinstance(model.fc, torch.nn.Identity)
