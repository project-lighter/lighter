#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`
#* Docker variables
IMAGE := lighter
VERSION := latest

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
	PYTHONPATH=$(PYTHONPATH) uv run pytest -c pyproject.toml

.PHONY: check-codestyle
check-codestyle:
	uvx isort --diff --check-only --settings-path pyproject.toml ./
	uvx black --diff --check --config pyproject.toml ./
	uvx pylint lighter

.PHONY: mypy
mypy:
	uvx mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	uvx safety check
	uvx bandit -ll --recursive lighter tests

.PHONY: lint
lint: test check-codestyle mypy check-safety