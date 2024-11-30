### Variables
# Define shell and Python environment variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

### Poetry Setup
# Install Poetry and configure environment
.PHONY: setup
setup:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -
	@echo "export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring" >> ~/.bashrc 
	@echo "export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring" >> ~/.profile
	poetry self add poetry-bumpversion@latest poetry-plugin-export@latest

### Installation
# Install project dependencies
.PHONY: install
install:
	poetry install -n

# Install pre-commit hooks
.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

### Linting & Testing
# Check code formatting and style
.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
	poetry run pylint lighter

# Run security checks
.PHONY: check-safety
check-safety:
	poetry check
	poetry export | poetry run safety check --stdin
	poetry run bandit -ll --recursive lighter tests

# Run tests with coverage report
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=lighter tests/

# Generate coverage badge
.PHONY: coverage
coverage:
	poetry run coverage-badge -o assets/images/coverage.svg -f

### Dependency Management
# Update development dependencies
.PHONY: update-dev-deps
update-dev-deps:
	poetry add -G dev bandit@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest \
		pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest \
		pytest-html@latest pytest-cov@latest
	poetry add -G dev --allow-prereleases black@latest

# Update main project dependencies
.PHONY: update-deps
update-deps:
	poetry add torch@latest torchvision@latest pytorch_lightning@latest torchmetrics@latest monai@latest

### Cleaning
# Remove temporary files and build artifacts
.PHONY: clean
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$|.DS_Store|.mypy_cache|.pytest_cache|.ipynb_checkpoints)" | xargs rm -rf
	rm -rf build/
