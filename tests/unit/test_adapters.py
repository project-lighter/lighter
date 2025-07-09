"""Unit tests for the adapters in lighter/adapters.py"""

import pytest
import torch
from torch import Tensor

from lighter.adapters import (
    BatchAdapter,
    CriterionAdapter,
    LoggingAdapter,
    MetricsAdapter,
    _ArgumentsAdapter,
    _ArgumentsAndTransformsAdapter,
    _TransformsAdapter,
)


def create_tensor(shape=(1,), value=None):
    """Helper function to create dummy tensors."""
    if value is not None:
        return Tensor([value])
    return torch.rand(*shape)


class TestTransformsAdapter:
    def test_no_transforms(self):
        """Test that adapter works correctly with no transforms."""
        adapter = _TransformsAdapter()
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        transformed_input, transformed_target, transformed_pred = adapter(input, target, pred)
        assert transformed_input.equal(input)
        assert transformed_target.equal(target)
        assert transformed_pred.equal(pred)

    def test_single_transform(self):
        """Test adapter with a single transform for each input."""
        transform = lambda x: x + 1  # noqa: E731
        adapter = _TransformsAdapter(input_transforms=transform, target_transforms=transform, pred_transforms=transform)
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        transformed_input, transformed_target, transformed_pred = adapter(input, target, pred)
        assert transformed_input.equal(input + 1)
        assert transformed_target.equal(target + 1)
        assert transformed_pred.equal(pred + 1)

    def test_multiple_transforms(self):
        """Test adapter with multiple transforms that are applied in sequence."""
        transform1 = lambda x: x * 2  # noqa: E731
        transform2 = lambda x: x - 1  # noqa: E731
        adapter = _TransformsAdapter(
            input_transforms=[transform1, transform2],
            target_transforms=[transform1, transform2],
            pred_transforms=[transform1, transform2],
        )
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        transformed_input, transformed_target, transformed_pred = adapter(input, target, pred)
        assert transformed_input.equal((input * 2) - 1)
        assert transformed_target.equal((target * 2) - 1)
        assert transformed_pred.equal((pred * 2) - 1)

    def test_invalid_transform(self):
        """Test that adapter raises ValueError for non-callable transforms."""
        adapter = _TransformsAdapter(input_transforms="not_a_callable")
        with pytest.raises(ValueError, match="Invalid transform type"):
            adapter(create_tensor(), create_tensor(), create_tensor())


class TestArgumentsAdapter:
    def test_positional_arguments(self):
        """Test adapter with all positional arguments."""
        adapter = _ArgumentsAdapter(input_argument=0, target_argument=1, pred_argument=2)
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        args, kwargs = adapter(input, target, pred)
        assert args == [input, target, pred]
        assert kwargs == {}

    def test_keyword_arguments(self):
        """Test adapter with all keyword arguments."""
        adapter = _ArgumentsAdapter(input_argument="in", target_argument="tar", pred_argument="pre")
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        args, kwargs = adapter(input, target, pred)
        assert args == []
        assert kwargs == {"in": input, "tar": target, "pre": pred}

    def test_mixed_arguments(self):
        """Test adapter with mixed positional and keyword arguments."""
        adapter = _ArgumentsAdapter(input_argument=0, target_argument="tar", pred_argument=1)
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        args, kwargs = adapter(input, target, pred)
        assert args == [input, pred]
        assert kwargs == {"tar": target}

    def test_invalid_positional_arguments(self):
        """Test that adapter validates consecutive positional arguments."""
        # Test for non-consecutive positional arguments
        with pytest.raises(ValueError, match="Positional arguments must be consecutive integers starting from 0"):
            _ArgumentsAdapter(input_argument=0, target_argument=2)

        # Test for non-zero starting positional argument
        with pytest.raises(ValueError, match="Positional arguments must be consecutive integers starting from 0"):
            _ArgumentsAdapter(input_argument=1, target_argument=2)


class TestArgumentsAndTransformsAdapter:
    def test_valid_initialization(self):
        """Test that adapter initializes correctly with valid arguments."""
        adapter = _ArgumentsAndTransformsAdapter(input_argument=0, target_argument=1, pred_argument=2)
        assert adapter.input_argument == 0
        assert adapter.target_argument == 1
        assert adapter.pred_argument == 2

    def test_invalid_initialization(self):
        """Test that adapter raises appropriate errors for invalid initialization."""
        with pytest.raises(ValueError, match="Input transforms provided but input_argument is None"):
            _ArgumentsAndTransformsAdapter(input_transforms=lambda x: x)
        with pytest.raises(ValueError, match="Target transforms provided but target_argument is None"):
            _ArgumentsAndTransformsAdapter(target_transforms=lambda x: x)
        with pytest.raises(ValueError, match="Pred transforms provided but pred_argument is None"):
            _ArgumentsAndTransformsAdapter(pred_transforms=lambda x: x)

    def test_call_with_transforms_and_arguments(self):
        """Test adapter correctly applies transforms and arranges arguments."""

        def mock_fn(pred, target, input):
            return pred + target + input

        adapter = _ArgumentsAndTransformsAdapter(
            pred_argument=0,
            target_argument=1,
            input_argument=2,
            input_transforms=lambda x: x * 2,
            target_transforms=lambda x: x + 1,
            pred_transforms=lambda x: x - 1,
        )
        input = create_tensor(value=1.0)
        target = create_tensor(value=2.0)
        pred = create_tensor(value=3.0)

        # Calculate expected result: (pred-1) + (target+1) + (input*2)
        # (3-1) + (2+1) + (1*2) = 2 + 3 + 2 = 7
        expected = Tensor([7.0])

        result = adapter(mock_fn, input, target, pred)
        assert result.equal(expected)

    def test_call_with_only_arguments(self):
        """Test adapter works correctly with only argument adaptation."""

        def mock_fn(a, b, c):
            return a, b, c

        adapter = _ArgumentsAndTransformsAdapter(input_argument="a", target_argument="b", pred_argument="c")
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        result = adapter(mock_fn, input, target, pred)
        assert result[0].equal(input)
        assert result[1].equal(target)
        assert result[2].equal(pred)

    def test_call_with_only_transforms(self):
        """Test adapter works correctly with only transforms."""

        def mock_fn(x):
            return x

        adapter = _ArgumentsAndTransformsAdapter(input_argument=0, input_transforms=lambda x: x * 2)
        input = create_tensor()
        result = adapter(mock_fn, input, None, None)
        assert result.equal(input * 2)


class TestBatchAdapter:
    def test_list_access(self):
        """Test adapter correctly accesses list elements."""
        adapter = BatchAdapter(input_accessor=0, target_accessor=1, identifier_accessor=2)
        batch = [create_tensor(), create_tensor(), "id"]
        input, target, identifier = adapter(batch)
        assert input.equal(batch[0])
        assert target.equal(batch[1])
        assert identifier == "id"

    def test_dict_access(self):
        """Test adapter correctly accesses dictionary elements."""
        adapter = BatchAdapter(input_accessor="in", target_accessor="tar", identifier_accessor="id")
        batch = {"in": create_tensor(), "tar": create_tensor(), "id": "identifier"}
        input, target, identifier = adapter(batch)
        assert input.equal(batch["in"])
        assert target.equal(batch["tar"])
        assert identifier == "identifier"

    def test_callable_access(self):
        """Test adapter correctly uses callable accessors."""
        adapter = BatchAdapter(input_accessor=lambda b: b["data"], target_accessor=lambda b: b["label"])
        batch = {"data": create_tensor(), "label": create_tensor()}
        input, target, identifier = adapter(batch)
        assert input.equal(batch["data"])
        assert target.equal(batch["label"])
        assert identifier is None

    def test_invalid_access(self):
        """Test adapter handles invalid access attempts appropriately."""
        # Test KeyError for non-existent dictionary key
        adapter = BatchAdapter(input_accessor="invalid_key")
        batch = {"in": create_tensor()}
        with pytest.raises(KeyError):
            adapter(batch)

        # Test ValueError for invalid accessor type
        adapter = BatchAdapter(input_accessor=0)
        batch = "invalid_batch_type"
        with pytest.raises(ValueError, match="Invalid accessor"):
            adapter(batch)


class TestCriterionAdapter:
    def test_call_with_arguments_and_transforms(self):
        """Test CriterionAdapter correctly applies transforms and arranges arguments."""

        def mock_criterion(pred, target, input):
            return pred + target + input

        adapter = CriterionAdapter(
            pred_argument=0,
            target_argument=1,
            input_argument=2,
            input_transforms=lambda x: x * 2,
            target_transforms=lambda x: x + 1,
            pred_transforms=lambda x: x - 1,
        )
        input = create_tensor(value=1.0)
        target = create_tensor(value=2.0)
        pred = create_tensor(value=3.0)

        # Calculate expected result: (pred-1) + (target+1) + (input*2)
        # (3-1) + (2+1) + (1*2) = 2 + 3 + 2 = 7
        expected = Tensor([7.0])

        result = adapter(mock_criterion, input, target, pred)
        assert result.equal(expected)


class TestMetricsAdapter:
    def test_call_with_arguments_and_transforms(self):
        """Test MetricsAdapter correctly applies transforms and arranges arguments."""

        def mock_metric(pred, target, input):
            return pred + target + input

        adapter = MetricsAdapter(
            pred_argument=0,
            target_argument=1,
            input_argument=2,
            input_transforms=lambda x: x * 2,
            target_transforms=lambda x: x + 1,
            pred_transforms=lambda x: x - 1,
        )
        input = create_tensor(value=1.0)
        target = create_tensor(value=2.0)
        pred = create_tensor(value=3.0)

        # Calculate expected result: (pred-1) + (target+1) + (input*2)
        # (3-1) + (2+1) + (1*2) = 2 + 3 + 2 = 7
        expected = Tensor([7.0])

        result = adapter(mock_metric, input, target, pred)
        assert result.equal(expected)


class TestLoggingAdapter:
    def test_no_transforms(self):
        """Test LoggingAdapter works correctly with no transforms."""
        adapter = LoggingAdapter()
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        transformed_input, transformed_target, transformed_pred = adapter(input, target, pred)
        assert transformed_input.equal(input)
        assert transformed_target.equal(target)
        assert transformed_pred.equal(pred)

    def test_single_transform(self):
        """Test LoggingAdapter works correctly with single transform."""
        transform = lambda x: x + 1  # noqa: E731
        adapter = LoggingAdapter(input_transforms=transform, target_transforms=transform, pred_transforms=transform)
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        transformed_input, transformed_target, transformed_pred = adapter(input, target, pred)
        assert transformed_input.equal(input + 1)
        assert transformed_target.equal(target + 1)
        assert transformed_pred.equal(pred + 1)

    def test_multiple_transforms(self):
        """Test LoggingAdapter works correctly with multiple transforms."""
        transform1 = lambda x: x * 2  # noqa: E731
        transform2 = lambda x: x - 1  # noqa: E731

        adapter = LoggingAdapter(
            input_transforms=[transform1, transform2],
            target_transforms=[transform1, transform2],
            pred_transforms=[transform1, transform2],
        )
        input, target, pred = create_tensor(), create_tensor(), create_tensor()
        transformed_input, transformed_target, transformed_pred = adapter(input, target, pred)
        assert transformed_input.equal((input * 2) - 1)
        assert transformed_target.equal((target * 2) - 1)
        assert transformed_pred.equal((pred * 2) - 1)
