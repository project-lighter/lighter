---
title: Troubleshooting
---

# Troubleshooting

Two switches help with any problem:

- `--print_config` prints the composed config, after all files and overrides, without building anything.
- `LIGHTER_DEBUG=1 lighter fit config.yaml` shows the full traceback, framework frames included.

Below is what Lighter prints for common mistakes. A test makes each mistake in a small project and checks that
the output is exactly this. The project is `proj/`, with `__lighter__.py`, a `model.py` that defines
`TinyModel(network, learning_rate)` and a `configs/config.yaml` that builds it. `<tmp>` stands for the folder
that holds `proj/`.

## Lighter cannot find your class

Your classes are `project.<file>.<Class>`, found through `__lighter__.py` ([Your code](your-code.md)).

```text
--8<-- "tests/integration/mistakes/missing_marker.txt"
```

```text
--8<-- "tests/integration/mistakes/target_without_project_prefix.txt"
```

A misspelled name, of your own class or an installed one, gets the closest match:

```text
--8<-- "tests/integration/mistakes/project_target_typo.txt"
```

## A key or value is wrong

An argument that the class does not take fails when the object is built. In a YAML file the arrow points at
the line; for an override, the last line names it:

```text
--8<-- "tests/integration/mistakes/override_typo.txt"
```

A misspelled top-level key is not an argument of anything, so it is added without an error. Check with
`--print_config`:

```text
--8<-- "tests/integration/mistakes/override_new_top_level_key.txt"
```

A reference must name an existing key, and an expression sees only Python's builtins and the modules in
`_imports_` (here, `_imports_: {math: math}` is missing):

```text
--8<-- "tests/integration/mistakes/bad_reference.txt"
```

```text
--8<-- "tests/integration/mistakes/bad_expression.txt"
```

`LighterModule` takes one metric or a `MetricCollection` per stage:

```text
--8<-- "tests/integration/mistakes/metrics_as_a_list.txt"
```

`trainer:` and `data:` are built from their `_target_`:

```text
--8<-- "tests/integration/mistakes/trainer_without_target.txt"
```

A YAML syntax error names the line where the parser stopped (here, line 9 is indented by one space instead of
two):

```text
--8<-- "tests/integration/mistakes/yaml_syntax_error.txt"
```

## The command line is wrong

```text
--8<-- "tests/integration/mistakes/missing_file.txt"
```

```text
--8<-- "tests/integration/mistakes/files_separated_by_comma.txt"
```

```text
--8<-- "tests/integration/mistakes/forgotten_equals.txt"
```

## Your code raises

An error raised in your own code while the run is built shows your frames only, and Lighter exits with 1:

```text
--8<-- "tests/integration/mistakes/bug_in_user_init.txt"
```

Errors raised later, during the stage (for example in `training_step`), are Lightning's and show the full
traceback.
