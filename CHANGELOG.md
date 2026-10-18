# Changelog

All notable changes to Lighter are listed here. Breaking changes come with a one-line migration.

## Unreleased

### Fixed

- `numpy` is no longer pinned below 2.0, so Lighter installs next to NumPy 2.
- Sparkwheel is capped at `<0.1` until Lighter is tested against its next release; the minimum is 0.0.11.
- `pyyaml` is declared as a dependency (Lighter imports it directly).
- Project URLs point at `github.com/project-lighter/lighter`.
