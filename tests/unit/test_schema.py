import pytest
from lighter.utils.runner import parse_config

@pytest.mark.parametrize("config", [
    "./projects/cifar10/experiments/monai_bundle_prototype.yaml",
    "./projects/cifar10/experiments/monai_bundle_prototype.yaml,./tests/configs/test1.yaml"
])
def test_config_schema_validation(config: str):
    """
    Test the validation of configuration schemas.

    This test ensures that the provided configuration files are parsed correctly
    and conform to the expected schema defined by ConfigSchema.

    Args:
        config (str): Path to the configuration file(s) to be validated.
    """
    # Parse the configuration file(s)
    parsed_config = parse_config(config=config.split(','))
    
    # Ensure the parsed configuration is a dictionary
    assert isinstance(parsed_config.config, dict)
