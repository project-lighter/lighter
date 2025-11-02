"""Unit tests for the Flow class."""

import torch
from torch import nn

from lighter.flow import Flow
from lighter.utils.types.enums import Data


class DummyModel(nn.Module):
    """Simple model that returns its input."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Identity()

    def forward(self, x):
        return self.linear(x)


def test_flow_model_single_string_arg():
    """Test that Flow can handle a single string as a model argument."""
    flow = Flow(
        batch={"input_data": "data"},
        model="input_data",  # Single string argument
        output={Data.PRED: Data.PRED},
    )

    model = DummyModel()
    batch = {"data": torch.randn(1, 10)}
    context = flow._unpack_batch(batch)
    context = flow._run_model(context, model)

    assert Data.PRED in context
    assert torch.equal(context[Data.PRED], batch["data"])


def test_flow_model_list_arg():
    """Test that Flow can handle a list as a model argument."""
    flow = Flow(
        batch={"input_data": "data"},
        model=["input_data"],  # List argument
        output={Data.PRED: Data.PRED},
    )

    model = DummyModel()
    batch = {"data": torch.randn(1, 10)}
    context = flow._unpack_batch(batch)
    context = flow._run_model(context, model)

    assert Data.PRED in context
    assert torch.equal(context[Data.PRED], batch["data"])


def test_flow_model_dict_arg():
    """Test that Flow can handle a dict as a model argument."""
    flow = Flow(
        batch={"input_data": "data"},
        model={"x": "input_data"},  # Dict argument
        output={Data.PRED: Data.PRED},
    )

    model = DummyModel()
    batch = {"data": torch.randn(1, 10)}
    context = flow._unpack_batch(batch)
    context = flow._run_model(context, model)

    assert Data.PRED in context
    assert torch.equal(context[Data.PRED], batch["data"])
