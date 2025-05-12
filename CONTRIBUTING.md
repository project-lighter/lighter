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
uv sync
```

3. Install pre-commit hooks:
```bash
uvx pre-commit install
```

## Codestyle

After installation you may execute code formatting:

```bash
make check-codestyle
```

To fix this run,
```bash
make codestyle
```

### Checks
Ensure that `check-codestyle` passes.


### Tests
Ensure all tests pass when running

```bash
make test
```


### Before submitting

Before submitting your code:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Format all your code
```bash
make codestyle
```
5. Run all checks:
```bash
make lint
```

## Other help

You can contribute by:
- Spreading the word about this library
- Writing short articles/tutorials about your use cases
- Sharing best practices and examples


# Documentation contribution
Our documentation is built using mkdocs and mkdocs-material. API reference is generated from docstrings using mkdocstrings.

## Serving the documentation locally

```bash
make docs
```

## Deploying the documentation

Documentation is automatically deployed when changes are merged to main.
