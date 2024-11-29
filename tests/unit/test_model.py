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
    system = DummySystem()
    replace_layer_with_identity(system.model, "fc")
    assert isinstance(system.model.fc, torch.nn.Identity)
