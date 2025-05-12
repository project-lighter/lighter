### Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

# Install uv so we can do everything with it
.PHONY: setup
setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: install
install:
	uv sync

#* Formatters
.PHONY: codestyle
codestyle:
	uvx pyupgrade --exit-zero-even-if-changed --py310-plus **/*.py
	uvx isort --settings-path pyproject.toml ./
	uvx black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Tests + Coverage
.PHONY: test
test:
	uv run pytest -c pyproject.toml --cov-report=html --cov=lighter tests/
	$(MAKE) coverage

.PHONY: coverage
coverage:
	uvx coverage-badge -o assets/images/coverage.svg -f

#* Linting checks
.PHONY: check-codestyle
check-codestyle:
	uvx isort --diff --check-only --settings-path pyproject.toml ./
	uvx black --diff --check --config pyproject.toml ./
	uv run pylint lighter

.PHONY: mypy
mypy:
	uvx mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	uvx safety check
	uvx bandit -ll --recursive lighter tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

#* Version bumps (through poetry-bumpversion because uv doesn't have version bumping yet)
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

.PHONY: docs
docs:
	uv run --group docs mkdocs serve
