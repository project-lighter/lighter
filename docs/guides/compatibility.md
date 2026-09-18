# Compatibility and local paired installation

The current working pair is **Lighter 0.2.0.dev0** and **Sparkwheel 0.1.0.dev0**. Lighter requires `sparkwheel>=0.1.0.dev0,<0.2.0` because its Runner uses retained definitions and isolated resolution scopes. Installing an earlier Sparkwheel release is not a supported workaround. These development changes have not been published by this work.

## Exact validation profiles

| Profile | Python | Torch / TorchVision | Lightning | TorchMetrics | NumPy | pandas |
|---|---|---|---|---|---|---|
| `reference` | 3.11 | 2.7.1 / 0.22.1 | 2.5.1 | 1.9.0 | 1.26.4 | 2.2.3 |
| `numpy2` | 3.11 | 2.7.1 / 0.22.1 | 2.6.1 | 1.9.0 | 2.2.6 | 2.2.3 |

The corresponding constraints live in `requirements/profiles/`. Core source tests and the download-free public workflow have run on these tuples. The NumPy 2 profile passed the integrated non-slow suite, so the obsolete `numpy<2` restriction is replaced by `numpy>=1.26.4,<3`. Every report must still identify the exact source revisions tested; a passing earlier checkout does not qualify a later patch automatically.

The strongest complete workflow evidence is CPU execution. Separate bounded checks exercised two-rank CPU/Gloo training and prediction, plus MPS float32 managed/native update parity. These do not certify CUDA, mixed precision, arbitrary distributed strategies, distributed checkpoint continuation or sharded construction. Metadata allows Python 3.10 and later; the qualified local profiles above use Python 3.11. Unexecuted OS/Python/version combinations remain unqualified.

Package dependency floors match the tested framework versions. They are resolver compatibility constraints, not an exhaustive support matrix. Profile files pin the main dependencies; they are not full transitive locks. The paired installation script resolves and hashes the complete installed environment for its interpreter/platform, preserving that result with its evidence.

## Build and check the unpublished pair

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and obtain both reviewed checkouts. From the Lighter checkout:

```bash
python3.11 scripts/check_paired_install.py \
  --sparkwheel /path/to/sparkwheel \
  --python /path/to/python3.11 \
  --profile reference \
  --output /path/to/project-lighter/artifacts/reference-pair
```

Use `--dry-run` first to print the exact source identities and paths without building. `--environment /another/new/path` selects an explicit environment directory. Both the output and environment must be new and outside both source checkouts. The script never overwrites an existing environment, tags, pushes or publishes anything.

It builds a wheel and source archive for each package, records their SHA-256 hashes and source Git identities, creates a fresh virtual environment, compiles `requirements.txt` with hashes, installs the pair and runs `uv pip check`. It removes inherited `PYTHONPATH`, uses isolated imports to verify that both packages come from the new environment, and checks both `lighter` and `python -m lighter` entrypoints.

It then copies the canonical example outside the source checkout and executes actual CLI fit/test/predict/resume processes. Numerical checkpoint/prediction oracles, preserved sample IDs, native restored LR and progress, static inspection, run listing, record reads and record comparison must pass. `installation.json`, command logs, the generated lock, copied example and workflow artifacts remain in the output directory. A failure remains a failed report with its command logs.

The generated hash lock names the local wheel files and is tied to that artifact location and platform. Retain the entire artifact directory. If relocating it, update only the local wheel URLs while preserving their hashes, or resolve again against those exact retained wheel files and compare dependency versions. A profile alone does not freeze package indexes or future transitive resolution. Offline installation additionally requires caching or retaining every dependency distribution; the script does not claim to produce an offline wheelhouse.

Build dependencies are constrained separately by `requirements/profiles/build.constraints`. uv's [dependency-source model](https://docs.astral.sh/uv/concepts/projects/dependencies/) distinguishes published dependency metadata from development sources; no absolute local source path is embedded in Lighter's published metadata.

## Editable development

For interactive development, install both checkouts into the same isolated environment. For example, with an already created Python 3.11 environment:

```bash
uv pip install --python /path/to/environment/bin/python \
  --constraint requirements/profiles/reference.constraints \
  --editable /path/to/sparkwheel --editable /path/to/lighter
```

This is convenient source development, not installed-wheel qualification. The paired verifier must still pass before a release. Avoid leaving an unrelated `PYTHONPATH` set when checking which packages are imported.

## Why there is no current registry-only uv.lock

The old lock selected Sparkwheel 0.0.x and stale framework dependencies. Retaining it after requiring the new API would falsely imply a reproducible installation. It has been removed. Until the matching Sparkwheel development distribution is available from the configured package registry, a registry-only `uv sync` cannot resolve this pair. The local script supplies the exact built Sparkwheel wheel instead of silently downgrading the requirement.

The shared CI setup now checks registry availability of the required Sparkwheel version before dependency installation and explains the paired workflow on failure. Availability is not runtime qualification. No unpushed branch is fetched, and unavailable dependency jobs are not reported as passing. Format/lint jobs that do not install the runtime can still run. Existing remote matrix definitions describe intended checks; they are not evidence that those environments passed for this unpublished pair.

When publishing is explicitly authorized later:

1. Pin and verify the final local source pair, including both exact profiles and installed artifacts.
2. Publish the compatible Sparkwheel distribution first and verify that registry resolution finds that artifact.
3. Regenerate Lighter's registry-only lock with `uv lock`, review dependency changes, and verify `uv sync --locked` against the intended profiles and Python versions. uv documents the distinction between [locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
4. Build and verify Lighter from that final commit, then release it through the ordinary review process. Development versions are distinct from final versions under the [Python version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/).

No publication, Git tag or push is performed by the current implementation work. The existing repository release workflow is separate and must only be triggered after that explicit release decision.

## Repository verification

With the matching pair installed, run:

```bash
python -m pytest tests -m "not slow"
python -m coverage run -m pytest tests -m "not slow"
python -m coverage combine
python -m coverage report
```

The non-slow suite includes actual two-process CPU/Gloo training and native/Lighter prediction controls when the backend is available. These preserve the mathematical update, sampler and CSV identity checks used during independent evaluation. They need local process creation and loopback communication; a platform without Gloo skips them explicitly.

Coverage includes the real CLI subprocesses and spawned training ranks using Coverage.py's documented [process collection](https://coverage.readthedocs.io/en/latest/subprocess.html). Combine process files before reporting; the existing 95% gate remains in place. Lint, repository formatting and `mypy src` are separate checks. A coverage percentage is an execution measure, not a correctness or hardware certification.
