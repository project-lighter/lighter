"""Importing a library must preserve host logging and multiprocessing policy."""

import os
import subprocess
import sys


def test_import_preserves_host_process_policies():
    code = r"""
import logging
import warnings
import sys
from multiprocessing import reduction
import pytorch_lightning
import sparkwheel
from loguru import logger
handler = logging.NullHandler()
logging.getLogger().handlers = [handler]
warning_hook = warnings.showwarning
exception_hook = sys.excepthook
pickler_dump = reduction.dump
loguru_handlers = dict(logger._core.handlers)
finders = list(sys.meta_path)
import lighter
assert logging.getLogger().handlers == [handler], "root logging handlers replaced"
assert warnings.showwarning is warning_hook, "warning hook replaced"
assert sys.excepthook is exception_hook, "exception hook replaced"
assert reduction.dump is pickler_dump, "multiprocessing serialization replaced"
assert logger._core.handlers == loguru_handlers, "application loguru sinks replaced"
assert sys.meta_path == finders, "dynamic importer installed without a project"
"""
    result = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
