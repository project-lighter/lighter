"""Explicit legacy logging setup remains usable without mutating the test host."""

import subprocess
import sys


def test_explicit_logging_setup_routes_standard_logs_custom_levels_and_warnings():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
import warnings
from loguru import logger
from lighter.utils.logging import _setup_logging
_setup_logging()
logger.info("direct_marker")
logging.getLogger("external").warning("standard_marker")
logging.getLogger("external").log(35, "custom_level_marker")
warnings.warn("warning_marker")
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    for marker in ("direct_marker", "standard_marker", "custom_level_marker", "warning_marker"):
        assert marker in output
    assert "Logging error" not in output
