---
title: Recipes
---

# Recipes

Things you do once a project runs. Each one is plain Lightning, configured from YAML. A test runs every file on
this page and covers the effect of every command.

## Train on several GPUs

```bash
lighter fit config.yaml trainer::accelerator=gpu trainer::devices=4 trainer::strategy=ddp
```

Lightning starts one process per device. Each builds the run from the same config, and only the first prints the
banner and saves the config. `batch_size` is per process. A test runs this with two CPU processes
(`trainer::accelerator=cpu trainer::devices=2`).

## Seed a run

```yaml
seed: 42
```

Without `seed:`, the seed is the `PL_GLOBAL_SEED` environment variable, else 0. Lighter passes it to Lightning's
`seed_everything(seed, workers=True)`, which seeds Python, NumPy, torch and the DataLoader workers, and
writes it into the saved config. Change it for one run with an override:

```bash
lighter fit config.yaml seed=7
```

For results that are identical on a GPU too, also set `trainer::deterministic=true`.
