# Compatibility and installation

The current development pair is **Lighter 0.2.0.dev0 / Sparkwheel 0.1.0.dev0**. Both sources are public. Older PyPI Sparkwheel releases do not provide the APIs this Lighter version requires, so install both explicit source revisions together.

## Install the current source pair

Use Python 3.11, Git and a new working directory. This **immutable reviewed snapshot** is Lighter `12c09fc9f85f5ff95ac41634939a29da5c9b87c5` with Sparkwheel `b73e786e8716d11a77206fb3482d21a621a4ba81`. It includes the current runtime and documentation corrections; a commit pin does not follow subsequent branch changes. Release handoffs name the exact pair they qualify.

On macOS or Linux, create a fresh environment and install the two distributions in one resolver invocation:

```bash
mkdir lighter-start
cd lighter-start
python3.11 -m venv .venv-lighter
. .venv-lighter/bin/activate
python -m pip install "pip==26.0"
LIGHTER_REV=12c09fc9f85f5ff95ac41634939a29da5c9b87c5
SPARKWHEEL_REV=b73e786e8716d11a77206fb3482d21a621a4ba81
python -m pip install \
  --constraint "https://raw.githubusercontent.com/project-lighter/lighter/$LIGHTER_REV/requirements/profiles/reference.constraints" \
  --build-constraint "https://raw.githubusercontent.com/project-lighter/lighter/$LIGHTER_REV/requirements/profiles/build.constraints" \
  "sparkwheel @ git+https://github.com/project-lighter/sparkwheel.git@$SPARKWHEEL_REV" \
  "lighter @ git+https://github.com/project-lighter/lighter.git@$LIGHTER_REV"
python -m pip check
python -c "import lighter, sparkwheel; print(lighter.__version__, lighter.__file__); print(sparkwheel.__version__, sparkwheel.__file__)"
```

Expect versions `0.2.0.dev0` and `0.1.0.dev0`, with imports inside `.venv-lighter`'s `site-packages`. Installation needs network access but no GitHub account. The constraints select the reference dependencies and build backend; they are not a complete transitive or offline lock. A fresh environment avoids reusing a different already-installed development snapshot. pip's installation metadata retains both VCS commit IDs.

The examples live in the repository, outside the installed wheel. In the same `lighter-start` directory, with the environment still active, fetch the matching example files:

```bash
git clone https://github.com/project-lighter/lighter.git lighter-examples
git -C lighter-examples checkout --detach "$LIGHTER_REV"
```

Now follow the [quick start](../quickstart.md). Installing and fetching examples are ordinary pip/Git operations; neither needs a supplied private checkout. Windows users can use `py -3.11` and the venv's PowerShell activation script, adapting the shell variable syntax. The literal commands above are POSIX shell commands.

For an RTX 5080/CUDA 12.8 qualification, install the official `torch==2.7.1+cu128` and `torchvision==0.22.1+cu128` wheels first. Keep those exact local versions in an additional constraints file passed to the **same pair-install command**, together with the reference constraints. Check versions, import origins and `python -m pip check` afterward. The CPU/macOS reference profile does not establish CUDA success; use the separately qualified accelerator handoff for the complete GPU commands and evidence.

<a id="editable-development"></a>
### Editable development

To edit the frameworks themselves, use the public sibling clones and editable installation in [Contributing](https://github.com/project-lighter/lighter/blob/854418a15478a7b35e0f885e3883218fe2d728a7/CONTRIBUTING.md). A wheel/VCS user install does not follow changes in a local checkout. Do not substitute registry-only `pip install lighter` or `uv sync` for the explicit development pair.

## Exact validation profiles

| Profile | Python | Torch / TorchVision | Lightning | TorchMetrics | NumPy | pandas |
|---|---|---|---|---|---|---|
| `reference` | 3.11 | 2.7.1 / 0.22.1 | 2.5.1 | 1.9.0 | 1.26.4 | 2.2.3 |
| `numpy2` | 3.11 | 2.7.1 / 0.22.1 | 2.6.1 | 1.9.0 | 2.2.6 | 2.2.3 |

Constraints are in `requirements/profiles/`. Core tests and the download-free workflow have run on these tuples at their recorded revisions. Passing evidence for one checkout does not qualify a later change automatically.

The strongest complete-workflow evidence is CPU execution. Separate bounded checks exercised two-rank CPU/Gloo training and prediction, plus MPS float32 managed/native update parity. These do not certify CUDA, mixed precision, arbitrary strategies, distributed continuation or sharded construction. Package metadata allows Python 3.10 and later; the local profiles above use 3.11. Other combinations require their own checks.

Dependency bounds describe resolver compatibility, not every possible supported combination. Profile files pin main dependencies; the paired verifier resolves and hashes its complete installed environment.

## Build and check the unpublished pair

This section is for maintainers qualifying artifacts. It is not required to start the example after the source-revision installation above.

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

After separate authorization, `just bump <part>` updates metadata and creates its configured local commit and tag. It does not publish the dependency or regenerate the registry lock. Remote publication requires a tag push named exactly `vMAJOR.MINOR.PATCH`, matching both package metadata and the source version, at a commit on fetched `main`. Development/prerelease tags and mismatches are rejected. Manual dispatch builds and retains workflow artifacts without publishing. The GitHub Release workflow applies the same stable-tag guard. Local preparation does not authorize a push or publication.
