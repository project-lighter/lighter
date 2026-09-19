"""Reject release tags unless stable metadata and fetched main ancestry agree.

Run with Python 3.11+ after fetching origin/main. This script does not publish.
"""

import argparse
import ast
import re
import subprocess
from pathlib import Path

STABLE_TAG = re.compile(r"refs/tags/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def git(root: Path, *arguments: str) -> str:
    """Read an identity from the checked-out repository."""
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True, stderr=subprocess.PIPE).strip()


def validate_release(root: Path, event: str, ref: str, sha: str) -> str:
    """Return the stable version only for a matching checkout on fetched main."""
    import tomllib

    if event != "push" or STABLE_TAG.fullmatch(ref) is None:
        raise ValueError("Publication requires a push of an exact stable vMAJOR.MINOR.PATCH tag.")
    version = ref.removeprefix("refs/tags/v")
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if project["version"] != version:
        raise ValueError("Release tag does not match project.version.")
    module = root / "src" / project["name"].replace("-", "_") / "__init__.py"
    source_versions = [
        ast.literal_eval(node.value)
        for node in ast.parse(module.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    if source_versions != [version]:
        raise ValueError("Release tag does not match the single source __version__ literal.")
    commit = git(root, "rev-parse", "--verify", f"{sha}^{{commit}}")
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError("Release checkout does not match the event commit.")
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
        check=True,
        capture_output=True,
        text=True,
    )
    return version


def main() -> None:
    """Check explicit event inputs; a nonzero exit prevents release steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--event", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        version = validate_release(args.root, args.event, args.ref, args.sha)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Release rejected: {error}\n")
    print(f"Verified stable release {version}: metadata, source version, checkout and main ancestry agree.")


if __name__ == "__main__":
    main()
