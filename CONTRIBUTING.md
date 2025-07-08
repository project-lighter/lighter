1. [Code contribution](#code-contribution)
    - [Dependencies](#dependencies)
    - [Codestyle](#codestyle)
      - [Checks](#checks)
        - [Before submitting](#before-submitting)
    - [Other help](#other-help)

2. [Documentation contribution](#documentation-contribution)
    - [Dependencies](#dependencies-1)
    - [Serving the documentation locally](#serving-the-documentation-locally)
    - [Deploying the documentation to GitHub Pages](#deploying-the-documentation-to-github-pages)


# Code contribution

## Dependencies

We use `uv` for fast Python package management. To set up the development environment:

1. Create and activate a virtual environment:
```bash
uv venv .venv
source .venv/bin/activate
```

2. Install development dependencies:
```bash
uv sync --dev
```

3. Install pre-commit hooks:
```bash
uvx pre-commit install
```

## Codestyle

After installation you may execute code formatting:

```bash
just format
```

### Checks
Ensure that `just format`, `just lint`, and `just types` pass.


### Tests
Ensure all tests pass when running

```bash
just test
```


### Before submitting

Before submitting your code:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Format all your code
```bash
just format
```
5. Run all checks:
```bash
just lint
just types
```

## Version Management and Releases

This project uses `bump-my-version` for versioning and Git tags for releases, similar to [TorchIO](https://github.com/fepegar/torchio).

**Versioning:**
The version is managed in `pyproject.toml` and `src/lighter/__init__.py`. To update the version, use `bump-my-version` via `just`:

```bash
just bump-version <patch|minor|major>
```
For example, to bump a patch version:
```bash
just bump-version patch
```
This command will automatically:
1.  Update the `version` in `pyproject.toml`.
2.  Update the `__version__` in `src/lighter/__init__.py`.
3.  Create a Git commit with the version bump.
4.  Create a Git tag (e.g., `v0.1.1`) corresponding to the new version.

**Automated Release Process:**
Upon pushing a Git tag (e.g., `git push origin v0.1.1`), a GitHub Actions workflow automatically:
1.  Builds the package using `uv build`.
2.  Publishes the package to PyPI using `uv publish`.

**Important:**
*   For PyPI publishing to work, ensure the `PYPI_TOKEN` secret is configured in the GitHub repository settings.
*   This setup does *not* automatically create a GitHub Release. You will need to create GitHub Releases manually if desired.

## Other help

You can contribute by:
- Spreading the word about this library
- Writing short articles/tutorials about your use cases
- Sharing best practices and examples


# Documentation contribution
Our documentation is built using mkdocs and mkdocs-material. API reference is generated from docstrings using mkdocstrings.

## Serving the documentation locally

```bash
just docs
```

## Deploying the documentation

Documentation is automatically deployed when changes are merged to main.
