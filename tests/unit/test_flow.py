"""Unit tests for the Flow class."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torchmetrics import MetricCollection

from lighter.flow import Flow
from lighter.utils.types.enums import Data, Mode


@pytest.fixture
def mock_model():
    """Returns a mock model."""
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.randn(2, 2)
    return model


@pytest.fixture
def mock_criterion():
    """Returns a mock criterion."""
    criterion = MagicMock(spec=nn.Module)
    criterion.return_value = torch.tensor(0.5)
    return criterion


@pytest.fixture
def mock_metrics():
    """Returns a mock metric collection."""
    metrics = MagicMock(spec=MetricCollection)
    metrics.update = MagicMock()
    return metrics


def test_flow_init():
    """Test Flow initialization."""
    flow = Flow(batch=["input", "target"])
    assert flow.batch_config == ["input", "target"]
    assert flow.model_config == {}
    assert flow.criterion_config == {}
    assert flow.metrics_config == {}
    assert flow.output_config == {}
    assert flow.logging_config == {}


def test_unpack_batch():
    """Test _unpack_batch method."""
    # Test with a list
    flow = Flow(batch=["input", "target"])
    batch = (torch.randn(2, 4), torch.randint(0, 2, (2,)))
    context = flow._unpack_batch(batch)
    assert torch.equal(context["input"], batch[0])
    assert torch.equal(context["target"], batch[1])

    # Test with a dict
    flow = Flow(batch={"input": 0, "target": 1})
    context = flow._unpack_batch(batch)
    assert torch.equal(context["input"], batch[0])
    assert torch.equal(context["target"], batch[1])


def test_run_model(mock_model):
    """Test _run_model method."""
    flow = Flow(batch=[], model=["input"])
    context = {"input": torch.randn(2, 4)}
    context = flow._run_model(context, mock_model)
    mock_model.assert_called_once_with(context["input"])
    assert Data.PRED in context


def test_run_criterion(mock_criterion):
    """Test _run_criterion method."""
    flow = Flow(batch=[], criterion=["pred", "target"])
    context = {"pred": torch.randn(2, 2), "target": torch.randint(0, 2, (2,))}
    context = flow._run_criterion(context, mock_criterion)
    mock_criterion.assert_called_once_with(context["pred"], context["target"])
    assert Data.LOSS in context


def test_run_metrics(mock_metrics):
    """Test _run_metrics method."""
    flow = Flow(batch=[], metrics=["pred", "target"])
    context = {"pred": torch.randn(2, 2), "target": torch.randint(0, 2, (2,))}
    context = flow._run_metrics(context, mock_metrics)
    mock_metrics.update.assert_called_once_with(context["pred"], context["target"])
    assert Data.METRICS in context


def test_apply_logging_transforms():
    """Test _apply_logging_transforms method."""
    flow = Flow(batch=[], logging={"transformed_pred": "pred"})
    context = {"pred": torch.randn(2, 2)}
    final_context = flow._apply_logging_transforms(context)
    assert "transformed_pred" in final_context
    assert torch.equal(final_context["transformed_pred"], context["pred"])


def test_build_output():
    """Test _build_output method."""
    flow = Flow(batch=[], output={"loss": "loss", "pred": "pred"})
    context = {"loss": torch.tensor(0.5), "pred": torch.randn(2, 2)}
    output = flow._build_output(context)
    assert "loss" in output
    assert "pred" in output
    assert torch.equal(output["loss"], context["loss"])
    assert torch.equal(output["pred"], context["pred"])


def test_get_default():
    """Test get_default method."""
    for mode in [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT]:
        flow = Flow.get_default(mode)
        assert isinstance(flow, Flow)


def test_call(mock_model, mock_criterion, mock_metrics):
    """Test __call__ method."""
    flow = Flow.get_default(Mode.TRAIN)
    batch = (torch.randn(2, 4), torch.randint(0, 2, (2,)))
    output = flow(batch, mock_model, mock_criterion, mock_metrics)
    assert Data.LOSS in output
    assert Data.PRED in output


def test_get_value():
    """Test _get_value method."""
    flow = Flow(batch=[])
    context = {
        "a": 1,
        "b": {"c": 2},
        "d": lambda ctx: ctx["a"] + 10,
    }

    # Test simple key
    assert flow._get_value(context, "a") == 1

    # Test nested key
    assert flow._get_value(context, "b.c") == 2

    # Test callable
    assert flow._get_value(context, "d") == 11

    # Test pipeline
    assert flow._get_value(context, ["a", lambda v: v + 1, lambda v: v * 2]) == 4

    # Test missing key
    with pytest.raises(KeyError):
        flow._get_value(context, "z")

    # Test missing nested key
    with pytest.raises(KeyError):
        flow._get_value(context, "b.z")

    # Test invalid key type
    with pytest.raises(TypeError):
        flow._get_value(context, 123)
