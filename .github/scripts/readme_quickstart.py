"""Run the README's quick start as written, with this commit's wheel and checkout in place of PyPI and GitHub.

Usage: <clean venv>/bin/python .github/scripts/readme_quickstart.py <wheel> <empty folder>

The commands run in the folder, with the venv's `pip` and `lighter`. Then `lighter --help` is timed: it must
answer without importing torch.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[2]
HELP_SECONDS = 2.0  # about 0.1 s on a laptop; CI machines are slower and noisier


def readme_commands() -> list[list[str]]:
    block = re.search(r"\[start:try\] -->\n```bash\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.S)
    assert block, "README.md has no quick start block between `--8<-- [start:try]` and `[end:try]`"
    return [shlex.split(line) for line in block[1].splitlines()]


def main(wheel: Path, folder: Path) -> None:
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]  # the venv's commands
    folder.mkdir(parents=True, exist_ok=True)
    for args in readme_commands():
        print("$", shlex.join(args), flush=True)
        if args == ["pip", "install", "lighter"]:
            args = [sys.executable, "-m", "pip", "install", str(wheel)]
        elif args[:2] == ["git", "clone"]:
            ignore = shutil.ignore_patterns(".git", ".venv", "dist", "lightning_logs", ".datasets")
            shutil.copytree(ROOT, folder / args[-1].rsplit("/", 1)[-1], ignore=ignore)
            continue
        subprocess.run([shutil.which(args[0]) or args[0], *args[1:]], cwd=folder, check=True)
    assert (folder / "lightning_logs" / "version_0" / "config.yaml").is_file(), "the run saved no config"

    lighter = shutil.which("lighter")
    times = []
    for _ in range(3):
        start = time.perf_counter()
        subprocess.run([lighter, "--help"], check=True, capture_output=True)
        times.append(time.perf_counter() - start)
    report = f"`lighter --help` answered in {min(times):.2f} s (fastest of 3)."
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as file:
            file.write(report + "\n")
    assert min(times) < HELP_SECONDS, f"`lighter --help` should answer in under {HELP_SECONDS} s"


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
