"""The docs stay short, and every code block in them is checked.

- A ```yaml block is composed like a config file and every top-level key is built.
- A ```python block runs as it is.
- Both run in a copy of examples/quickstart, so they may name `project.model.Regressor` or load `config.yaml`.
- A block that includes a file (`--8<--`) is skipped here: the file is run by the test of its own folder.
- A `lighter` command in a shell block must parse with the CLI's own parser.
- Floats are written with a dot: YAML 1.1 reads `1e-3` as a string.
"""

import re
import shlex
import shutil
import sys
from pathlib import Path

import pytest

import lighter.engine.runner as runner
from lighter.engine.runner import load_config
from lighter.utils.dynamic_imports import import_module_from_path

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md"))
LINE_BUDGET = 3000
FENCE = re.compile(r"^( *)```(\w*)[^\n]*\n(.*?)^\1```", re.M | re.S)
INCLUDE = re.compile(r'--8<-- "([^"]+)"')
# Folders whose files are run by their own tests (examples, projects) or are test outputs (golden files).
TESTED = ("examples/", "projects/tabular_regression/", "tests/", "README.md")
EXPONENT_FLOAT = re.compile(r"(?<![\w.])\d+e[-+]?\d+(?![\w.])")


def blocks(language: str) -> list:
    found = []
    for page in PAGES:
        for match in FENCE.finditer(page.read_text()):
            indent, lang, code = match.groups()
            if lang == language:
                code = "\n".join(line[len(indent) :] for line in code.splitlines())
                line = page.read_text()[: match.start()].count("\n") + 1
                found.append(pytest.param(code, id=f"{page.relative_to(DOCS)}:{line}"))
    return found


@pytest.fixture
def quickstart(tmp_path, monkeypatch):
    """A copy of examples/quickstart as the current folder, imported as `project`."""
    folder = tmp_path / "quickstart"
    shutil.copytree(ROOT / "examples" / "quickstart", folder)
    monkeypatch.chdir(folder)
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    import_module_from_path("project", folder)
    yield folder
    drop()


def test_docs_stay_within_the_line_budget():
    lines = sum(len(page.read_text().splitlines()) for page in PAGES)
    assert lines <= LINE_BUDGET, f"docs/ has {lines} lines of Markdown, more than {LINE_BUDGET}. Delete something first."


def test_included_files_come_from_tested_folders():
    for page in PAGES:
        for path in INCLUDE.findall(page.read_text()):
            assert path.startswith(TESTED), f"{page.name} includes {path}, which no test runs"


@pytest.mark.parametrize("code", blocks("yaml"))
def test_yaml_blocks_build(code, quickstart):
    if INCLUDE.search(code):
        return
    assert not EXPONENT_FLOAT.search(code), "write floats with a dot, e.g. 0.001: YAML 1.1 reads 1e-3 as a string"
    (quickstart / "block.yaml").write_text(code)
    config = load_config("block.yaml")
    for key in list(config.get()):
        if key != "_imports_":
            config.resolve(key)


@pytest.mark.parametrize("code", blocks("python"))
def test_python_blocks_run(code, quickstart):
    if INCLUDE.search(code):
        return
    exec(compile(code, "<docs>", "exec"), {"__name__": "__docs__"})


class Parsed(BaseException):
    """Raised instead of building the run: the command line parsed."""


@pytest.mark.parametrize("code", [*blocks("bash"), *blocks("console")])
def test_lighter_commands_parse(code, monkeypatch):
    monkeypatch.setattr(runner, "build", lambda *inputs: (_ for _ in ()).throw(Parsed()))
    monkeypatch.setattr(runner, "load_config", lambda *inputs: (_ for _ in ()).throw(Parsed()))
    for line in code.replace("\\\n", " ").splitlines():
        line = line.strip().removeprefix("$ ")
        if not line.startswith(("lighter ", "python -m lighter ")):
            continue
        assert not EXPONENT_FLOAT.search(line), f"write floats with a dot: {line}"
        args = shlex.split(line.removeprefix("python -m "))[1:]
        assert not any(arg.endswith((".yaml", ".yml")) and "," in arg for arg in args), line
        monkeypatch.setattr(sys, "argv", ["lighter", *args])
        with pytest.raises((Parsed, SystemExit)) as raised:
            runner.cli()
        assert raised.type is Parsed or "--help" in args or "--version" in args, f"`{line}` does not parse"
