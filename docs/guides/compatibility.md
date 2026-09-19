# Compatibility and installation

This checkout is the unpublished development pair **Lighter 0.2.0.dev0 / Sparkwheel 0.1.0.dev0**. Lighter requires `sparkwheel>=0.1.0.dev0,<0.2.0` for retained definitions and isolated resolution scopes. An older published Sparkwheel package cannot replace the supplied current source.

## Install the current source pair

Use Python 3.11. The commands below assume you have received both source checkouts as siblings in this layout:

```text
project-lighter/
├── lighter/
└── sparkwheel/
```

Start in the directory containing `project-lighter`. The environment directory must be new. These commands install both local distributions in one resolver invocation; dependency downloads may need network access.

```bash
cd project-lighter
python3.11 -m venv .venv-lighter
. .venv-lighter/bin/activate
python -m pip install --constraint lighter/requirements/profiles/reference.constraints --editable ./sparkwheel --editable ./lighter
python -m pip check
python -c "import lighter, sparkwheel; print(lighter.__file__); print(sparkwheel.__file__)"
```

The final command should name these two supplied source trees. If it names another checkout, inspect your environment and `PYTHONPATH` before proceeding. Keep this environment active and follow the [quick start](../quickstart.md), starting from the current `project-lighter` directory.

<a id="editable-development"></a>
This editable installation is the ordinary source-development route. It is distinct from the built-wheel qualification below. The profile constrains dependencies; it is not a complete transitive or offline lock. Do not substitute registry-only `pip install lighter` or `uv sync` for this unpublished pair.

## Exact validation profiles

| Profile | Python | Torch / TorchVision | Lightning | TorchMetrics | NumPy | pandas |
|---|---|---|---|---|---|---|
| `reference` | 3.11 | 2.7.1 / 0.22.1 | 2.5.1 | 1.9.0 | 1.26.4 | 2.2.3 |
| `numpy2` | 3.11 | 2.7.1 / 0.22.1 | 2.6.1 | 1.9.0 | 2.2.6 | 2.2.3 |

Constraints are in `requirements/profiles/`. Core tests and the download-free workflow have run on these tuples at their recorded revisions. Passing evidence for one checkout does not qualify a later change automatically.

The strongest complete-workflow evidence is CPU execution. Separate bounded checks exercised two-rank CPU/Gloo training and prediction, plus MPS float32 managed/native update parity. These do not certify CUDA, mixed precision, arbitrary strategies, distributed continuation or sharded construction. Package metadata allows Python 3.10 and later; the local profiles above use 3.11. Other combinations require their own checks.

Dependency bounds describe resolver compatibility, not every possible supported combination. Profile files pin main dependencies; the paired verifier resolves and hashes its complete installed environment.

## Build and check the unpublished pair

This section is for maintainers qualifying artifacts. It is not required to start the example after editable installation.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). From the Lighter checkout, the existing script builds and installs both packages into a fresh environment, then executes the diagnostic outside the checkout:

```bash
python scripts/check_paired_install.py --help
```

Provide `--sparkwheel` with the sibling source path, `--python` with the chosen Python 3.11 executable, `--profile reference` (or `numpy2`) and `--output` with a new artifact directory outside both checkouts. `--dry-run` prints identities and paths without building; `--environment` selects a new environment directory. The script never overwrites an environment, tags, pushes or publishes packages.

It preserves wheels/source archives, hashes, source identities, a generated dependency lock, import-origin checks, `installation.json`, logs and copied-example artifacts. It checks fit/test/predict/resume, exact prediction IDs/values, restored LR/progress, static inspection and records. This diagnostic qualification alone does not qualify Compare and Continue or another scientific project. A failed command remains a failed report.

The generated hash lock refers to retained local wheels and its platform. Keep the artifact directory. Relocation requires updating local wheel URLs without changing hashes, or resolving again against those same wheels and comparing versions. Offline installation additionally needs retained dependency distributions; this script does not create an offline wheelhouse. Build constraints live separately in `requirements/profiles/build.constraints`.

## Why there is no current registry-only uv.lock

The old lock selected incompatible Sparkwheel 0.0.x and has been removed. A registry-only resolution cannot reproduce this pair until a compatible distribution is available in that registry. The local paired workflow supplies the source or built wheel explicitly.

Main, full-matrix, documentation and other default CI callers check the required registry version before runtime installation. Availability is not qualification; unavailable dependency jobs are not passing jobs.

For this unpublished pair, PR type/test jobs use a temporary, explicit Sparkwheel commit pinned in `.github/workflows/ci.yml`. The setup action checks out that exact commit from the public companion repository, verifies its identity, and installs both sources together with the development group and existing reference/build constraints. The job summary records the actual tested Lighter checkout separately from its PR head, companion commit, versions and import origins. The resolved environment is logged and included in the ordinary test artifact as `paired-ci-requirements.txt`; this is not an offline lock.

Paired jobs use `uv run --no-sync` so later commands consume that installed environment. Their existing Python 3.12/Ubuntu checks and coverage gate remain in place; the Python 3.11 local profile does not certify that combination. A changed companion requires updating the full commit pin and checking the new pair. Format/lint jobs do not receive the companion checkout. PR jobs receive no Codecov secret; coverage calculation still runs.

This PR-only source route does not establish registry delivery or make default main/deployment jobs pass. Remote matrix definitions describe intended checks until actual results exist. Resolve the publication and registry-lock sequence below before merging for release readiness; a green paired PR is insufficient by itself.

When a release is separately authorized, qualify the exact pair, publish compatible Sparkwheel first, regenerate and verify Lighter's registry lock, then qualify and release Lighter. See uv's distinction between [locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/). No publication is part of local installation.

## Repository verification

With development test dependencies installed in the matching environment, maintainers run the repository's test, coverage, lint, formatting and type-check targets. The non-slow suite includes native/math controls and actual CPU/Gloo subprocesses where available; an unavailable backend is explicitly skipped.

Coverage collects CLI and spawned-process execution and combines process files before applying the existing gate. Coverage measures execution, not correctness or hardware support. The project testing and release procedures are separate from the short user journey.

## Version maintenance

Version recipes use `bump-my-version==0.30.1` through `uvx`. Supported parts are `major`, `minor`, `patch` and `release`.

```bash
just bump-dry release
```

This previews 0.2.0.dev0 → 0.2.0 without writes, commits or tags. A numeric `patch` from 0.2.0.dev0 instead previews 0.2.1; use `release` to promote the existing development version. A new development cycle needs an explicit version decision.

After separate authorization, `just bump <part>` updates metadata and creates its configured local commit and tag. It does not publish the dependency or regenerate the registry lock. Remote tag workflows may publish pushed tags; do not route development tags through them. Local preparation does not authorize a push or publication.
