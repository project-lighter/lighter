"""Build and verify a local Lighter/Sparkwheel pair in an isolated wheel install.

Requires Python 3.11+ for this script and uv on PATH. Nothing is published.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib


def git_identity(directory: Path) -> dict:
    def read(*arguments):
        result = subprocess.run(["git", "-C", str(directory), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"path": str(directory), "commit": read("rev-parse", "HEAD"), "status": read("status", "--porcelain")}


def python_in(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_record_packages(record: dict, environment: Path, versions: dict[str, str]) -> None:
    """Check the packages actually used by each non-isolated workflow child."""
    packages = record.get("environment", {}).get("packages", {})
    for name, expected_version in versions.items():
        observed = packages.get(name, {})
        filename = observed.get("module_path")
        path = Path(filename).resolve() if isinstance(filename, str) else None
        if path is None or not path.is_file() or environment.resolve() not in path.parents:
            raise ValueError(
                f"Attempt {record.get('attempt_id')} loaded {name} outside the installed environment: {filename!r}"
            )
        if observed.get("loaded_version") != expected_version or observed.get("distribution_version") != expected_version:
            raise ValueError(f"Attempt {record.get('attempt_id')} used inconsistent {name} versions: {observed!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparkwheel", required=True, type=Path, help="Sparkwheel source checkout")
    parser.add_argument("--lighter", type=Path, default=Path(__file__).resolve().parents[1], help="Lighter source checkout")
    parser.add_argument("--output", required=True, type=Path, help="New directory for artifacts, logs and the generated lock")
    parser.add_argument("--environment", type=Path, help="New venv directory (default: OUTPUT/environment)")
    parser.add_argument("--python", default=sys.executable, help="Python 3.11 interpreter for the qualified profiles")
    parser.add_argument("--profile", choices=("reference", "numpy2"), default="reference")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating files or running uv")
    options = parser.parse_args()
    source = options.lighter.resolve()
    sparkwheel = options.sparkwheel.resolve()
    output = options.output.resolve()
    environment = (options.environment or output / "environment").resolve()
    if output.exists() or environment.exists():
        parser.error("Choose new output and environment paths; this command never overwrites an existing environment.")
    for checkout in (source, sparkwheel):
        if any(path == checkout or checkout in path.parents for path in (output, environment)):
            parser.error(
                "Output and environment must be outside both source checkouts, so source imports cannot hide wheel defects."
            )
        if not (checkout / "pyproject.toml").is_file():
            parser.error(f"Missing pyproject.toml in {checkout}")
    profiles = source / "requirements/profiles"
    constraint = profiles / f"{options.profile}.constraints"
    if not constraint.is_file():
        parser.error(f"Missing profile {constraint}")
    versions = {
        name: tomllib.loads((path / "pyproject.toml").read_text())["project"]["version"]
        for name, path in (("lighter", source), ("sparkwheel", sparkwheel))
    }
    plan = {
        "sources": {"lighter": git_identity(source), "sparkwheel": git_identity(sparkwheel)},
        "versions": versions,
        "profile": options.profile,
        "constraints": str(constraint),
        "python": options.python,
        "output": str(output),
        "environment": str(environment),
        "steps": [
            "build both wheels and sdists",
            "resolve a hash-locked local pair",
            "install into fresh venv",
            "check installed imports and both CLIs",
            "copy and execute canonical workflow outside source",
            "verify inspection and completed run-record JSON",
        ],
    }
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        return
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv first; see https://docs.astral.sh/uv/getting-started/installation/")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    clean_environment = dict(os.environ)
    for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_SYSTEM_PYTHON"):
        clean_environment.pop(variable, None)
    clean_environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", UV_NO_PROGRESS="1")
    commands = []
    plan.update(status="running", commands=commands, inherited_pythonpath_removed=True)

    def save():
        (output / "installation.json").write_text(json.dumps(plan, indent=2) + "\n")

    def run(label, arguments, *, cwd=output):
        command = [str(argument) for argument in arguments]
        commands.append({"label": label, "arguments": command, "cwd": str(cwd)})
        save()
        result = subprocess.run(command, cwd=cwd, env=clean_environment, capture_output=True, text=True)
        (logs / f"{label}.stdout").write_text(result.stdout)
        (logs / f"{label}.stderr").write_text(result.stderr)
        commands[-1]["returncode"] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {logs / f'{label}.stderr'}")
        return result.stdout

    try:
        observed_python = json.loads(
            run(
                "python-profile",
                [options.python, "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
            )
        )
        if observed_python != [3, 11]:
            raise ValueError("These qualified profiles use Python 3.11. Other Python versions require separate qualification.")
        plan["uv_version"] = run("uv-version", [uv, "--version"]).strip()
        dist = output / "dist"
        dist.mkdir()
        for name, checkout in (("sparkwheel", sparkwheel), ("lighter", source)):
            run(
                f"build-{name}",
                [
                    uv,
                    "build",
                    "--wheel",
                    "--sdist",
                    "--python",
                    options.python,
                    "--build-constraints",
                    profiles / "build.constraints",
                    "--out-dir",
                    dist,
                    checkout,
                ],
            )
        artifacts = sorted(dist.iterdir())
        plan["artifacts"] = [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in artifacts
            if path.is_file()
        ]
        wheels = [next(dist.glob(f"{name}-{version}-*.whl")) for name, version in versions.items()]
        requirements = output / "pair.in"
        requirements.write_text(
            "\n".join(f"{name} @ {wheel.as_uri()}" for name, wheel in zip(versions, wheels, strict=True)) + "\n"
        )
        lock = output / "requirements.txt"
        run("create-environment", [uv, "venv", environment, "--python", options.python, "--no-python-downloads"])
        python = python_in(environment)
        run(
            "lock",
            [
                uv,
                "pip",
                "compile",
                requirements,
                "--constraint",
                constraint,
                "--generate-hashes",
                "--python",
                python,
                "--output-file",
                lock,
            ],
        )
        run("install", [uv, "pip", "install", "--python", python, "--require-hashes", "--requirement", lock])
        run("dependency-check", [uv, "pip", "check", "--python", python])
        installed = run(
            "installed-imports",
            [
                python,
                "-I",
                "-c",
                "import importlib.metadata as m,json,sys,lighter,sparkwheel; print(json.dumps({'prefix':sys.prefix,'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()},'imports':{'lighter':lighter.__file__,'sparkwheel':sparkwheel.__file__},'versions':{'lighter':lighter.__version__,'sparkwheel':sparkwheel.__version__}}))",
            ],
        )
        plan["installed"] = json.loads(installed)
        if plan["installed"]["versions"] != versions:
            raise ValueError("Installed package versions differ from the built pair")
        for imported in plan["installed"]["imports"].values():
            if environment not in Path(imported).resolve().parents:
                raise ValueError(f"Import escaped installed environment: {imported}")
        console = environment / ("Scripts/lighter.exe" if os.name == "nt" else "bin/lighter")
        run("console-cli", [console, "--help"])
        run("module-cli", [python, "-I", "-m", "lighter", "--help"])
        project = output / "reference-workflow"
        shutil.copytree(
            source / "projects/tabular_regression", project, ignore=shutil.ignore_patterns("__pycache__", "outputs", "*.pyc")
        )
        inspection = json.loads(
            run("inspect-json", [python, "-m", "lighter", "inspect", "config.yaml", "--json"], cwd=project)
        )
        if not isinstance(inspection, dict):
            raise ValueError("Inspection did not produce a JSON object")
        workflow_output = output / "workflow-output"
        run("workflow", [python, project / "workflow.py", "--output-dir", workflow_output], cwd=project)
        records = json.loads(
            run("runs-list-json", [console, "runs", "list", workflow_output / "lighter_runs", "--json"], cwd=project)
        )
        if len(records) != 4 or sorted(record["stage"] for record in records) != ["fit", "fit", "predict", "test"]:
            raise ValueError("Expected four fit/test/predict/resume records")
        if any(record["status"] != "completed" for record in records):
            raise ValueError("An installed workflow record did not complete")
        for record in records:
            verify_record_packages(record, environment, versions)
            run_path = workflow_output / "lighter_runs" / record["attempt_id"]
            shown = json.loads(
                run(
                    f"runs-show-{record['attempt_id']}",
                    [python, "-m", "lighter", "runs", "show", run_path],
                    cwd=project,
                )
            )
            if shown["attempt_id"] != record["attempt_id"]:
                raise ValueError("Run list/show identity mismatch")
        fits = sorted(
            (record for record in records if record["stage"] == "fit"),
            key=lambda record: record["observed_end"]["global_step"],
        )
        compared = json.loads(
            run(
                "runs-diff-json",
                [
                    console,
                    "runs",
                    "diff",
                    workflow_output / "lighter_runs" / fits[0]["attempt_id"],
                    workflow_output / "lighter_runs" / fits[1]["attempt_id"],
                ],
                cwd=project,
            )
        )
        if not isinstance(compared, list) or not any(
            change["path"] == "observed_end::global_step" and change["before"] == 9 and change["after"] == 15
            for change in compared
        ):
            raise ValueError("Run comparison did not report native step progression from 9 to 15")
        plan["workflow"] = json.loads((workflow_output / "workflow.json").read_text())
        plan["record_attempts"] = [record["attempt_id"] for record in records]
        plan["status"] = "passed"
        save()
        print(f"Installed paired-wheel verification passed: {output / 'installation.json'}")
    except BaseException as error:
        plan.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        save()
        raise


if __name__ == "__main__":
    main()
