### Variables
# Define shell and Python environment variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

# Install
.PHONY: setup
setup: 
	pip install uv
	
#* Installation
.PHONY: install
install:
	uv pip install -e .

#* Formatters
.PHONY: codestyle
codestyle:
	uvx pyupgrade --exit-zero-even-if-changed --py37-plus **/*.py
	uvx isort --settings-path pyproject.toml ./
	uvx black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test: 
	uv run pytest -c pyproject.toml 
	$(MAKE) coverage

.PHONY: coverage
coverage:
	uvx coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	uvx isort --diff --check-only --settings-path pyproject.toml ./
	uvx black --diff --check --config pyproject.toml ./
	uv run pylint lighter

.PHONY: bump-prerelease
bump-prerelease:
	uvx --with poetry-bumpversion poetry version prerelease

.PHONY: bump-patch
bump-patch:
	uvx --with poetry-bumpversion poetry version patch

.PHONY: bump-minor
bump-minor:
	uvx --with poetry-bumpversion poetry version minor

.PHONY: bump-major
bump-major:
	uvx --with poetry-bumpversion poetry version major

.PHONY: mypy
mypy:
	uvx mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	uvx safety check
	uvx bandit -ll --recursive lighter tests

.PHONY: lint
lint: test check-codestyle mypy check-safety
