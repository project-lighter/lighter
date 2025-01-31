"""
Logging utilities for configuring and setting up custom logging using Loguru and Rich.

This module provides functionality to set up visually appealing logs with custom formatting,
traceback handling, and suppression of detailed logs from specified modules. It includes color
mapping for different log levels and handlers to intercept and redirect logging to Loguru.
"""

import importlib

# List of modules to suppress in Rich traceback for cleaner output
SUPPRESSED_MODULES = [
    "fire",
    "monai.bundle",
    "pytorch_lightning.trainer",
    "lightning_utilities",
    "torch.utils.data.dataloader",
]


LOGGING_COLOR_MAP = {
    "TRACE": "dim blue",
    "DEBUG": "cyan",
    "INFO": "bold",
    "SUCCESS": "bold green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}


def _setup_logging():
    """
    Configures custom logging using Loguru and Rich for visually appealing logs.
    Sets up traceback handling and suppression of specific modules.
    Must be run before importing anything else to set up the loggers correctly.
    """
    import inspect
    import logging
    import warnings

    import rich.logging
    import rich.traceback
    from loguru import logger

    def formatter(record: dict) -> str:
        """Format log messages for better readability and clarity. Used to configure Loguru with a Rich handler."""
        lvl_name = record["level"].name
        lvl_color = LOGGING_COLOR_MAP.get(lvl_name, "cyan")
        return (
            "[not bold green]{time:YYYY/MM/DD HH:mm:ss.SSS}[/not bold green]  |  {level.icon}  "
            f"[{lvl_color}]{lvl_name:<10}[/{lvl_color}]|  [{lvl_color}]{{message}}[/{lvl_color}]"
        )

    class InterceptHandler(logging.Handler):
        """Handler to redirect other libraries' logging to Loguru. Taken from Loguru's documentation:
        https://github.com/Delgan/loguru?tab=readme-ov-file#entirely-compatible-with-standard-logging
        """

        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists.
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message.
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Intercept logging and redirect to Loguru. Must be called before importing other libraries to work.
    logging.getLogger().handlers = [InterceptHandler()]

    # Configure Rich traceback.
    suppress = [importlib.import_module(name) for name in SUPPRESSED_MODULES]
    rich.traceback.install(show_locals=False, width=120, suppress=suppress)
    # Rich handler for Loguru. Time and level are handled by the formatter.
    rich_handler = rich.logging.RichHandler(markup=True, show_time=False, show_level=False)
    logger.configure(handlers=[{"sink": rich_handler, "format": formatter}])

    # Capture the `warnings` standard module with Loguru
    # https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
    warnings.showwarning = lambda message, *args, **kwargs: logger.opt(depth=2).warning(message)
