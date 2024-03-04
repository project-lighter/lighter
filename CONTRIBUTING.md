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

We use `poetry` to manage the [dependencies](https://github.com/python-poetry/poetry).
If you dont have `poetry`, you should install with `make poetry-download`.

To install dependencies and prepare [`pre-commit`](https://pre-commit.com/) hooks you would need to run `install` command:

```bash
make install
make pre-commit-install
```

To activate your `virtualenv` run `poetry shell`.

## Codestyle

After installation you may execute code formatting.

```bash
make codestyle
```

### Checks

Many checks are configured for this project. Command `make check-codestyle` will check black and isort.
The `make check-safety` command will look at the security of your code.

Comand `make lint` applies all checks.

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Run `make codestyle` to format your changes.
1. Run `make lint` to ensure that types, security and docstrings are okay.

## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share your best practices with us.


# Documentation contribution
Our documentation is located in the `docs/` folder and is built using `mkdocs` and `mkdocs-material`.

The API reference is generated automatically from the docstrings in the code using `mkdocstrings`. Our docstrings follow the `google` style.

##  Dependencies
To install `mkdocs-material` together with the required dependencies run:

```bash
pip install mkdocs-material mkdocs-autorefs mkdocstrings mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

## Serving the documentation locally
While working on the documentation, you can serve it locally to see the changes in real-time.

```bash
cd docs/
mkdocs serve
```

## Deploying the documentation to GitHub Pages

The documentation is automatically deployed once the changes are merged into the `main` branch.
