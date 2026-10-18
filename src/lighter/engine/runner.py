"""Build a Lightning run from YAML configs and overrides, then run one stage (`lighter fit config.yaml`).

Torch and Lightning are imported only when a run is built, so `lighter --help` and `--print_config` are instant.
"""

import argparse
import copy
import difflib
import importlib.util
import inspect
import logging
import os
import pydoc
import re
import sys
import sysconfig
import traceback
import warnings
from pathlib import Path
from typing import Any

import yaml
from sparkwheel import BaseError, Config, TargetNotFoundError

from lighter import __version__
from lighter.utils.dynamic_imports import import_module_from_path

logger = logging.getLogger(__name__)

STAGES = {
    "fit": "Train a model",
    "validate": "Validate a model",
    "test": "Test a model",
    "predict": "Run predictions with a model",
}
HELP = """\
examples:
  lighter {stage} configs/train.yaml
  lighter {stage} configs/base.yaml configs/big.yaml trainer::max_epochs=10 model::optimizer::lr=0.001

Inputs are YAML files and overrides, applied left to right. Overrides:
  key::sub=value   set a value, written in YAML (10, 0.001, true, null, [1, 2], {{a: 1}})
  =key={{...}}       replace a whole section instead of merging into it
  ~key             delete a key
Quote overrides that hold $, @, spaces or brackets: 'model::lr=$@base_lr * 2'.
New keys belong in YAML: put them in a file and stack it, e.g. lighter {stage} base.yaml extra.yaml.
--print_config shows the composed config without running. LIGHTER_DEBUG=1 shows full tracebacks.
"""
# Frames from these folders are not the user's code.
FOLDERS = {sysconfig.get_paths()[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
FRAMEWORK = (*(folder + os.sep for folder in FOLDERS), str(Path(__file__).parents[1]) + os.sep, "<")


def _is_override(item: Any) -> bool:
    return isinstance(item, str) and not Path(item).is_file() and ("=" in item or item.startswith("~"))


def _config_dirs(inputs: tuple) -> list[Path]:
    return [Path(item).resolve().parent for item in inputs if isinstance(item, (str, Path)) and not _is_override(item)]


def _find_up(start: Path, name: str) -> Path | None:
    """The first of `start` and its parents that contains `name`."""
    return next((folder for folder in (start, *start.parents) if (folder / name).exists()), None)


def find_project_root(inputs: tuple) -> Path | None:
    """The folder holding `__lighter__.py`, searched upward from each config file's folder, then from the cwd."""
    roots = {root for folder in _config_dirs(inputs) if (root := _find_up(folder, "__lighter__.py"))}
    if len(roots) > 1:
        raise ValueError(f"The config files belong to different projects: {', '.join(map(str, sorted(roots)))}.")
    return roots.pop() if roots else _find_up(Path.cwd(), "__lighter__.py")


def _project_hint(error: TargetNotFoundError, inputs: tuple, root: Path | None) -> str | None:
    """Explain a `_target_` that names the user's own code without `project.` or without a `__lighter__.py`."""
    match = re.search(r"function: '([\w.]+)'", str(error))
    parts = match.group(1).lstrip(".").split(".") if match else ["project"]
    local = parts[1:] if parts[0] == "project" else parts
    if not local or importlib.util.find_spec(local[0]) is not None:  # empty, or an installed package
        return None
    if root is not None:
        if parts[0] != "project" and ((root / f"{local[0]}.py").exists() or (root / local[0]).is_dir()):
            return f"Code in your project folder is imported as `project`: did you mean 'project.{'.'.join(local)}'?"
        return None
    for folder in [*_config_dirs(inputs), Path.cwd()]:
        if found := _find_up(folder, f"{local[0]}.py") or _find_up(folder, local[0]):
            marker = found / "__lighter__.py"
            return f"To import your own code, create an empty {marker} and use '_target_: project.{'.'.join(local)}'."
    return None


def _lightning_package(obj: Any, kind: str, key: str) -> str:
    """Return which Lightning package (`pytorch_lightning` or `lightning.pytorch`) defines the `kind` that `obj` is."""
    for package in ("pytorch_lightning", "lightning.pytorch"):
        if package in sys.modules and isinstance(obj, getattr(sys.modules[package], kind)):
            return package
    hint = f" Add a `_target_` to `{key}:`." if isinstance(obj, dict) else ""
    raise TypeError(f"`{key}` must build a {kind}, got {type(obj).__name__}.{hint}")


def _missing_file(item: str) -> FileNotFoundError:
    path = Path(item)
    siblings = [file.name for file in path.parent.glob("*.y*ml")]
    if "," in item:
        hint = "Separate config files with spaces, not commas."
    elif path.suffix not in (".yaml", ".yml"):
        hint = f"Did you forget `=`? Overrides look like `{item}=value`."
    elif close := difflib.get_close_matches(path.name, siblings, n=1):
        hint = f"Did you mean {path.with_name(close[0])}?"
    else:
        hint = f"Paths are relative to the current directory, {Path.cwd()}."
    return FileNotFoundError(f"Config file not found: {item}\n  {hint}")


def load_config(*inputs: Any) -> Config:
    """Compose config files, dicts and overrides in order, on top of `seed: 0`.

    A string that names a file is a YAML file; other strings with `=` (or starting with `~`) are overrides.
    """
    config = Config().update({"seed": 0})
    for item in inputs:
        if isinstance(item, (str, Path)) and not _is_override(item):
            if not Path(item).is_file():
                raise _missing_file(str(item))
            item = str(Path(item).resolve())  # absolute, so Sparkwheel does not warn about `..` in the path
        config.update(item)
    return config


def to_yaml(config: dict) -> str:
    """The YAML of `--print_config` and of the saved config.yaml."""
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def build(*inputs: Any) -> tuple[Any, Any, Any]:
    """Build `(model, trainer, datamodule)` from config files, dicts and overrides without running a stage.

    Inputs are composed by `load_config`. The seed comes from the top-level `seed:` key (default: 0) and is
    set before anything is built.
    `datamodule` is None when the config has no `data:` key (the model's own dataloader hooks are then used).
    """
    config = load_config(*inputs)
    from pytorch_lightning import seed_everything  # after the config files are read, so their errors are instant
    from pytorch_lightning.utilities import rank_zero_only

    seed = config.resolve("seed") if isinstance(config.get("seed"), str) else config.get("seed")  # `@`, `%`, `$`
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"`seed` must be an integer, got {seed!r}.")
    config.set("seed", seed)
    try:
        saved = copy.deepcopy(config.get())  # before resolving, which drops `_imports_`
    except Exception:  # a dict config holding Python objects that cannot be copied
        saved = config.get()
    seed_everything(seed, workers=True)
    root = find_project_root(inputs)
    if root is not None:
        import_module_from_path("project", root)

    try:
        model = config.resolve("model")
        trainer = config.resolve("trainer")
        datamodule = None if config.get("data") is None else config.resolve("data")
    except TargetNotFoundError as error:
        hint = _project_hint(error, inputs, root)
        if hint is None:
            raise
        raise TargetNotFoundError(str(error).split("\n")[0], error.source_location, hint) from error
    packages = {_lightning_package(model, "LightningModule", "model"), _lightning_package(trainer, "Trainer", "trainer")}
    if datamodule is not None:
        packages.add(_lightning_package(datamodule, "LightningDataModule", "data"))
    if len(packages) > 1:
        raise TypeError(
            "The run mixes `pytorch_lightning` and `lightning.pytorch` objects, which Lightning rejects. Use one "
            "package throughout, e.g. `trainer::_target_: lightning.pytorch.Trainer` if your model imports `lightning`."
        )
    from lighter.engine.saved_config import SaveConfig

    trainer.callbacks.append(SaveConfig(saved))
    logs = trainer.log_dir if trainer.logger else "off (trainer::logger is false)"  # every rank asks: a broadcast
    if rank_zero_only.rank == 0:
        files = " ".join(str(item) for item in inputs if isinstance(item, (str, Path)) and not _is_override(item))
        configs = f"{files or 'dict'} + {sum(map(_is_override, inputs))} overrides"
        logger.info(f"Lighter {__version__} | project: {root or 'none'} | {configs} | seed: {seed}")
        logger.info(f"Logs: {logs or 'none'}")  # no folder under fast_dev_run
    return model, trainer, datamodule


class Runner:
    """Runs a Lightning stage from configuration inputs."""

    def run(self, stage: str, inputs: list, **stage_kwargs: Any) -> Any:
        """Build the run from `inputs` (see `build`) and return the result of `trainer.<stage>(...)`.

        Args:
            stage: "fit", "validate", "test" or "predict".
            inputs: Config file paths, dicts and overrides, applied in order.
            **stage_kwargs: Keyword arguments for the trainer stage method (e.g., ckpt_path, verbose).
        """
        if stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}', expected one of {list(STAGES)}.")
        model, trainer, datamodule = build(*inputs)
        _check_stage_data(stage, datamodule)
        return getattr(trainer, stage)(model, datamodule=datamodule, **stage_kwargs)


class _Stderr(logging.StreamHandler):
    """Logs to the current `sys.stderr`, like `logging.lastResort`."""

    stream = property(lambda self: sys.stderr, lambda self, stream: None)


def _check_stage_data(stage: str, datamodule: Any) -> None:
    """A LighterDataModule must have the dataloader that `stage` runs on."""
    name = {"fit": "train", "validate": "val", "test": "test", "predict": "predict"}[stage]
    if getattr(datamodule, f"{name}_dataloader", True) is None:
        raise ValueError(f"`lighter {stage}` needs `data::{name}_dataloader`, which is not set.")


def _short(path: str) -> str:
    return os.path.relpath(path) if path and Path(path).resolve().is_relative_to(Path.cwd()) else path


def _module_hint(error: ModuleNotFoundError, inputs: list) -> str | None:
    """Explain `from models import ...` in the user's code when `models` is a folder of their project."""
    try:
        root = find_project_root(tuple(inputs))
    except ValueError:
        return None
    top = (error.name or "").split(".")[0]
    if root is None or not top or top == "project" or not ((root / f"{top}.py").exists() or (root / top).is_dir()):
        return None
    return f"Code in your project folder is imported as `project`: write `from project.{error.name} import ...`."


def _report(error: BaseException, inputs: list) -> int:
    """Print an error raised while building the run, without framework frames. Returns the exit code.

    The first line is `file:line (key): Type: message` when the config location is known. A bug in the
    user's own code (the root cause was raised there) also shows those frames and exits with 1, like Python.
    """
    cause = error
    while (inner := cause.value if isinstance(cause, pydoc.ErrorDuringImport) else cause.__cause__) is not None:
        cause = inner  # pydoc.ErrorDuringImport holds the error raised while importing the user's module
    frames = [frame for frame in traceback.extract_tb(cause.__traceback__) if not frame.filename.startswith(FRAMEWORK)]
    message = getattr(error, "_original_message", str(error))  # Sparkwheel's message without its snippet
    suggestion = getattr(error, "suggestion", None)
    if isinstance(cause, ModuleNotFoundError) and not suggestion:
        suggestion = _module_hint(cause, inputs)
    where = error.source_location if isinstance(error, BaseError) else None
    lines = [f"{type(error).__name__}: {message}"]
    if frames:
        lines[:0] = ["Traceback (your code only, LIGHTER_DEBUG=1 shows all frames):", *traceback.format_list(frames)]
    if isinstance(error, yaml.MarkedYAMLError) and (mark := error.problem_mark):
        lines[-1] = f"{_short(mark.name)}:{mark.line + 1}: {type(error).__name__}: {error.problem}"
    elif where:
        key = f" ({where.id})" if where.id else ""
        place = f"{_short(where.filepath)}:{where.line}" if where.filepath else str(where)  # or an override
        lines[-1] = f"{place}{key}: {type(error).__name__}: {message}"
    if cause is not error and not (isinstance(cause, BaseError) or suggestion or str(cause) in message):
        lines.append(f"  From {type(cause).__name__}: {cause}")
    if where:
        lines.append(getattr(error, "_get_config_snippet", lambda: "")())
    if suggestion:
        lines.append(f"  {suggestion.replace('💡 ', '')}")
    keys = [(item, item.lstrip("=~").split("=")[0]) for item in inputs if _is_override(item)]
    if where and (overrides := [item for item, k in keys if where.id and (k == where.id or k.startswith(f"{where.id}::"))]):
        lines.append(f"  Overridden by: {' '.join(overrides)}")
    print("\n".join(line.rstrip() for line in lines if line), file=sys.stderr)
    return 1 if frames else 2


def cli() -> None:
    """Entry point for the lighter CLI."""
    parser = argparse.ArgumentParser(prog="lighter", description="Run a Lightning stage from YAML configs.")
    parser.add_argument("--version", action="version", version=f"lighter {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for stage, help_text in STAGES.items():
        stage_parser = subparsers.add_parser(
            stage,
            help=help_text,
            description=help_text,
            epilog=HELP.format(stage=stage),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        stage_parser.add_argument("inputs", nargs="+", help="Config files and overrides, e.g. config.yaml model::lr=0.001")
        stage_parser.add_argument(
            "--print_config", "--print-config", action="store_true", help="Print the composed config and exit."
        )
        stage_parser.add_argument(
            "--ckpt_path", help="A checkpoint file, or `last` for the last checkpoint in the checkpoint callback's `dirpath`."
        )
        stage_parser.add_argument(
            "--weights_only",
            action=argparse.BooleanOptionalAction,
            help="Restrict checkpoint loading to tensors (safer for untrusted files). Needs Lightning 2.6 or newer.",
        )
        if stage in ("validate", "test"):
            stage_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, help="Print results.")
        if stage == "predict":
            stage_parser.add_argument(
                "--return_predictions",
                action=argparse.BooleanOptionalAction,
                help="Keep predictions in memory and return them (Lightning's default unless spawning processes).",
            )

    args, extra = parser.parse_known_args()  # overrides may also follow options: `config.yaml --ckpt_path x a=1`
    if unknown := [item for item in extra if item.startswith("-")]:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.inputs += extra
    if args.ckpt_path == "best":
        parser.error("--ckpt_path best works only after `fit` in the same process. Pass the checkpoint file instead.")
    if not any(isinstance(handler, _Stderr) for handler in logging.getLogger("lighter").handlers):
        logging.getLogger("lighter").addHandler(_Stderr())
    logging.getLogger("lighter").setLevel(logging.INFO)
    stage_kwargs = {k: v for k, v in vars(args).items() if k not in ("command", "inputs", "print_config") and v is not None}
    try:  # errors in the config and in building objects; errors inside the stage propagate unchanged
        if args.print_config:
            print(to_yaml(load_config(*args.inputs).get()), end="")
            return
        model, trainer, datamodule = build(*args.inputs)
        _check_stage_data(args.command, datamodule)
        stage = getattr(trainer, args.command)
        if args.weights_only is not None and "weights_only" not in inspect.signature(stage).parameters:
            raise ValueError("--weights_only needs Lightning 2.6 or newer.")
    except Exception as error:
        if os.environ.get("LIGHTER_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            raise
        sys.exit(_report(error, args.inputs))
    if args.command != "fit" and args.ckpt_path is None and trainer.is_global_zero:
        logger.warning(f"No --ckpt_path: `{args.command}` uses the model as the config builds it, not a trained checkpoint.")
    with warnings.catch_warnings():
        if args.ckpt_path == "last" and trainer.world_size == 1:  # under DDP, stopping only rank 0 would hang the rest
            warnings.filterwarnings("error", message=r".*there is no last checkpoint available")
        try:
            stage(model, datamodule=datamodule, **stage_kwargs)
        except Warning as warning:
            if "no last checkpoint" not in str(warning):
                raise
            print(
                "`--ckpt_path last` found no last checkpoint: each run logs to a new version folder. Pass the "
                "checkpoint file instead, or give the checkpoint callback a fixed `dirpath`.",
                file=sys.stderr,
            )
            sys.exit(2)
    best = getattr(trainer.checkpoint_callback, "best_model_path", "") if args.command == "fit" else ""
    if best and trainer.is_global_zero:
        logger.info(f"Best checkpoint: {_short(best)}")
