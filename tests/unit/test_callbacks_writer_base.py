import pytest

from lighter.callbacks.writer.base import LighterBaseWriter


def test_writer_initialization():
    with pytest.raises(TypeError):
        LighterBaseWriter(path="test", writer="tensor")
