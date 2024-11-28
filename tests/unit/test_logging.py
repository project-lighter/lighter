import pytest
from lighter.utils.logging import _setup_logging

def test_setup_logging():
    _setup_logging()
    assert True  # Just ensure no exceptions are raised
