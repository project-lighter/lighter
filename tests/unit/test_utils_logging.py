import warnings
from unittest.mock import MagicMock, patch

from loguru import logger

from lighter.utils.logging import _setup_logging


def test_setup_logging():
    """Test basic logging setup."""
    _setup_logging()
    assert True  # Just ensure no exceptions are raised


def test_warnings_handler():
    """Test that warnings are properly captured by loguru."""
    _setup_logging()  # Setup logging to ensure warnings are captured

    # Create a mock logger with opt method
    mock_logger = MagicMock()
    mock_opt = MagicMock()
    mock_opt.warning = MagicMock()
    mock_logger.opt.return_value = mock_opt

    # Mock logger to verify warning is captured
    with patch.object(logger, "opt", mock_logger.opt):
        # Trigger a warning
        warnings.warn("Test warning")
        # Verify the warning was captured by loguru
        mock_logger.opt.assert_called_with(depth=2)
        mock_opt.warning.assert_called_once()
