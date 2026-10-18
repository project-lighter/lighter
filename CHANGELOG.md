# Changelog

All notable changes to Lighter are listed here. Breaking changes come with a one-line migration.

## Unreleased

### Changed

- The boolean CLI flags (`--weights_only`, `--verbose`, `--return_predictions`) also accept a `--no-` form.

### Removed

- `import lighter` no longer replaces the root logging handlers, installs a rich traceback hook or overrides
  `warnings.showwarning`. Lighter logs through the standard `logging` module (logger name `lighter`).
- LighterModule no longer logs optimizer statistics (`train/optimizer/...`).
  Migration: add `pytorch_lightning.callbacks.LearningRateMonitor` to `trainer::callbacks`.
- `lighter.utils.misc`, `lighter.utils.data` (`collate_replace_corrupted`), `lighter.utils.logging` and
  `lighter.utils.types.enums` are gone. Migration: pass stage names as strings, e.g. `Runner().run("fit", inputs)`.
- `writer_registry` is replaced by the `WRITERS` dict in `lighter.callbacks.file_writer`.
  Migration: pass a callable as `writer_fn` for a custom writer.
- Prediction writers no longer empty Lightning's private list of predictions after each batch.
  Migration: `lighter predict ... --no-return_predictions` keeps memory flat when the writer is all you need.

### Fixed

- `numpy` is no longer pinned below 2.0, so Lighter installs next to NumPy 2.
- Sparkwheel is capped at `<0.1` until Lighter is tested against its next release; the minimum is 0.0.11.
- `pyyaml` is declared as a dependency (Lighter imports it directly).
- Project URLs point at `github.com/project-lighter/lighter`.
