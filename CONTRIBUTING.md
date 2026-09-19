# Contributing to Lighter

Keep changes focused on a concrete workflow or contract. Scientific steps remain Python, configuration composes objects, and native Lightning owns execution. Include relevant tests and documentation with behavior changes.

## Set up the supplied pair

Follow [current-pair installation](docs/guides/compatibility.md#install-the-current-source-pair) first. This checkout depends on unpublished Sparkwheel APIs; `just setup` uses registry-based `uv sync` and is not the setup route for the supplied pair.

With that environment active, start in `project-lighter`, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then add the existing development group:

```bash
cd lighter
uv pip install --python ../.venv-lighter/bin/python --constraints requirements/profiles/reference.constraints --editable ../sparkwheel --editable . --group dev
python -m pip check
```

The `dev` group includes `doc`, `maintain`, `quality`, `types` and `test`, plus pre-commit tooling. It does not replace the requirement to supply both local packages. See `pyproject.toml` for the authoritative group definitions.

## Run relevant checks

From the Lighter checkout, with that environment active, invoke the installed tools directly:

```bash
python -m ruff check
python -m ruff format --check
python -m mypy src
python -m pytest tests -m "not slow"
python -m mkdocs build --strict
```

Choose focused tests while developing, then complete the checks appropriate to the change. The non-slow suite includes subprocess/native/math controls; unavailable platform capabilities remain explicit skips. Do not claim that a documentation build establishes scientific correctness.

For a local documentation preview, use `python -m mkdocs serve --dev-addr localhost:8000`. Existing `just` recipes remain in the repository, but many invoke `uv run` and may trigger registry resolution. Direct commands above use the environment containing the supplied pair.

## Source map

| Path | Responsibility |
|---|---|
| `src/lighter/model.py` | LighterModule steps, measurements and optimizer interface |
| `src/lighter/data.py` | Loader wrapper |
| `src/lighter/engine/` | Runner, managed construction, inspection and records |
| `src/lighter/callbacks/` | Writers and selected-parameter freezing |
| `src/lighter/utils/` | Supporting utilities |
| `tests/` | Unit and integration checks |
| `docs/`, `mkdocs.yml` | Manual pages, API rendering and navigation |
| `projects/` | Diagnostic, research walkthrough and labeled specialist references |

## Maintain the documentation

Keep one authoritative explanation of each contract and link it from related pages. Preserve useful page paths and incoming anchors when reorganizing. State working directory, prerequisites, literal command, expected artifact and next decision for runnable workflows. Distinguish complete examples from excerpts.

When changing a runnable example, execute the touched commands and check their real outputs. Use the existing diagnostic verifier when applicable. Build the site strictly and check links/anchors; review the rendered page for navigation and code readability. Keep source, profiles and evidence claims aligned. Generated internal entries do not make underscored helpers public extension APIs.

## Prepare a contribution or release

Describe the concrete problem, resulting behavior and validation in a pull request. Preserve unrelated changes and report platform or dependency limits. Do not update expected scientific results merely to hide a failure.

Release work follows [paired artifact qualification and ordering](docs/guides/compatibility.md#build-and-check-the-unpublished-pair): verify the final pair, publish compatible Sparkwheel first, then regenerate/verify Lighter's registry lock and release artifacts. Version and tag commands have side effects; consult [version maintenance](docs/guides/compatibility.md#version-maintenance) before using them. A documentation edit or local build is not a release operation.

Contributions use the project's [MIT license](LICENSE). Report reproducible issues through [GitHub](https://github.com/project-lighter/lighter/issues).
