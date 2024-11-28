import pytest
from lighter.utils.runner import parse_config

def test_parse_config_no_config():
    with pytest.raises(ValueError):
        parse_config()
