# Install uv so we can do everything with it
setup:
    curl -LsSf https://astral.sh/uv/install.sh | sh

install:
    uv sync

format:
    uvx tox -e format

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
    uv run --group docs mkdocs serve

cli-smoke-tests:
    uvx tox -e cli
