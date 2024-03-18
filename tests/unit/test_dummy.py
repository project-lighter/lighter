"""Tests for hello function."""

import pytest

from lighter.utils.misc import ensure_list


@pytest.mark.parametrize(
    ("input", "result"),
    [
        ([2, 3], [2, 3]),
        (1, [1]),
        ((2, 3), [2, 3]),
        (None, []),
    ],
)
def test_ensure_list(input, result):
    """Example test with parametrization."""
    assert ensure_list(input) == result
