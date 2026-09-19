"""Fail early when registry-only CI cannot obtain this branch's paired API."""

import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 CI, supplied by the setup action.
    import tomli as tomllib


metadata = tomllib.loads(Path("pyproject.toml").read_text())
requirement = next(value for value in metadata["project"]["dependencies"] if value.startswith("sparkwheel"))
result = subprocess.run(
    ["uv", "pip", "compile", "-", "--no-deps", "--python", sys.executable, "--no-header"],
    input=requirement + "\n",
    capture_output=True,
    text=True,
)
if result.returncode:
    print(result.stderr, file=sys.stderr)
    raise SystemExit(
        f"Registry-only CI cannot resolve {requirement}. The development pair may not be published yet. "
        "Use scripts/check_paired_install.py with explicit reviewed Lighter and Sparkwheel checkouts; "
        "see docs/guides/compatibility.md. Never lower the dependency to an older API to make CI pass."
    )
print(f"Registry has a candidate satisfying {requirement}; this availability check is not runtime qualification.")
