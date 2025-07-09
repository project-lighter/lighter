# Contributing to Lighter

Thank you for your interest in contributing to Lighter! This guide will help you get started.

## Quick Start

1. **Set up your development environment:**
    ```bash
    just setup
    ```
    This will install `uv` (if needed), sync all dependencies, and set up pre-commit hooks.

2. **Make your changes and test them:**
    ```bash
    just test        # Run tests
    just lint        # Check code style
    just types       # Run type checking
    ```

3. **Submit your contribution** via a pull request.

## Available Commands

We use [just](https://github.com/casey/just) as our command runner. Here are the available commands:

### Development Setup
- `just setup` - Complete development environment setup (installs uv, syncs dependencies, installs pre-commit)
- `just clean` - Clean up build artifacts, caches, and temporary files

### Code Quality
- `just lint` - Run linting checks
- `just types` - Run type checking with mypy
- `just test` - Run the test suite
- `just coverage` - Generate test coverage report and badge

### Documentation
- `just docs` - Serve documentation locally at http://localhost:8000

### Version Management
- `just bump [patch|minor|major]` - Bump version and create git tag (default: patch)
- `just bump-dry [patch|minor|major]` - Preview version bump without making changes
- `just push` - Push commits and tags to remote

## Development Workflow

### Making Changes

1. **Fork and clone** the repository
2. **Set up your environment:**
    ```bash
    just setup
    ```

3. **Create a feature branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```

4. **Make your changes** and add tests for new functionality

5. **Run quality checks:**
    ```bash
    just lint
    just types
    just test
    ```

6. **Commit your changes** (pre-commit hooks will run automatically)

7. **Push and create a pull request**

### Before Submitting a PR

Ensure all of these pass:
- ✅ `just lint` - Code follows style guidelines
- ✅ `just types` - No type checking errors
- ✅ `just test` - All tests pass
- ✅ Add tests for any new functionality
- ✅ Update documentation if needed

## Project Structure

```
src/lighter/           # Main package code
├── __init__.py       # Package initialization and version
├── engine/           # Core engine components
├── system.py         # System class
└── utils/            # Utility modules

tests/                # Test suite
docs/                 # Documentation source
```

## Release Process

Releases are automated:

1. **Bump version:** `just bump patch` (or `minor`/`major`)
2. **Push tags:** `just push`
3. **Automated publishing:** GitHub Actions will automatically build and publish to PyPI when a tag is pushed
