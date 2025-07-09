default:
    @just --list

clean:
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .tox
    rm -rf .venv
    rm -rf dist
    rm -rf **/__pycache__
    rm -rf src/*.egg-info
    rm -f .coverage
    rm -f coverage.*
    rm -f .coverage.*

@install_uv:
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

setup: install_uv
    uv sync --all-extras --all-groups
    uv run pre-commit install

lint:
    uvx tox -e lint

types:
    uvx tox -e types

test:
    uvx tox -e pytest

coverage:
    uvx tox -e coverage
    uvx coverage-badge -o assets/images/coverage.svg -f

docs:
    uv run --only-group doc mkdocs serve

bump part="patch":
    uvx bump-my-version bump {{part}} --verbose

bump-dry part="patch":
    uvx bump-my-version bump {{part}} --dry-run --verbose --allow-dirty

push:
    git push && git push --tags
