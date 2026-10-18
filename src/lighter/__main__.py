"""`python -m lighter` runs the Lighter CLI."""

import os
import sys

if __name__ == "__main__":
    # Like the `lighter` command, import your code through `__lighter__.py`, not from the current directory.
    sys.path = [path for path in sys.path if path not in ("", os.getcwd())]
    from lighter.engine.runner import cli

    cli()
