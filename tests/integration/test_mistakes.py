"""What `lighter fit` prints for common mistakes, compared with tests/integration/mistakes/<case>.txt.

After an intended change in the output, regenerate the expected files with `UPDATE_GOLDEN=1 pytest tests/integration`.
"""

import os
import re
import sys
from pathlib import Path

import pytest
import pytorch_lightning  # imported before capsys, so Lightning's own log lines stay out of the captured output

import lighter
from lighter.engine.runner import cli

GOLDEN = Path(__file__).parent / "mistakes"

MODEL = """import torch

from lighter import LighterModule


class TinyModel(LighterModule):
    def __init__(self, network, learning_rate=0.1):
        super().__init__(network=network)
        self.learning_rate = learning_rate


class BuggyModel(TinyModel):
    def __init__(self, network):
        super().__init__(network)
        self.head = torch.nn.Linear(network.out_feature, 2)
"""

CONFIG = """trainer:
  _target_: pytorch_lightning.Trainer
  logger: false

num_classes: 2

model:
  _target_: project.model.TinyModel
  learning_rate: 0.1
  network:
    _target_: torch.nn.Linear
    in_features: 4
    out_features: "@num_classes"
"""

METRICS = """trainer:
  _target_: pytorch_lightning.Trainer
  logger: false

model:
  _target_: lighter.LighterModule
  network:
    _target_: torch.nn.Linear
    in_features: 4
    out_features: 2
  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: binary
"""

# case: (folder to run from, arguments after `lighter fit`)
CASES = {
    "yaml_key_typo": ("proj", "configs/typo.yaml"),
    "override_typo": ("proj", "configs/config.yaml model::learnin_rate=0.01"),
    # Sparkwheel 0.0.11 accepts new keys from overrides silently; Sparkwheel 0.1 will report `seeed` as unknown.
    "override_new_top_level_key": ("proj", "configs/config.yaml seeed=1 --print_config"),
    "wrong_target": ("proj", "configs/config.yaml model::network::_target_=torch.nn.Lienar"),
    "project_target_typo": ("proj", "configs/config.yaml model::_target_=project.model.TinyModle"),
    "target_without_project_prefix": ("proj", "configs/config.yaml model::_target_=model.TinyModel"),
    "missing_marker": ("proj", "configs/config.yaml"),
    "run_from_parent_folder": (".", "proj/configs/config.yaml"),
    "run_from_config_folder": ("proj/configs", "config.yaml"),
    "missing_file": ("proj", "configs/confg.yaml"),
    "files_separated_by_comma": ("proj", "configs/config.yaml,configs/config.yaml"),
    "forgotten_equals": ("proj", "configs/config.yaml trainer::fast_dev_run"),
    "bad_reference": ("proj", "configs/config.yaml model::network::out_features=@num_clases"),
    "bad_expression": ("proj", "configs/config.yaml model::learning_rate=$math.sqrt(@num_classes)"),
    "metrics_as_a_list": ("proj", "configs/metrics.yaml"),
    "bug_in_user_init": ("proj", "configs/config.yaml model::_target_=project.model.BuggyModel ~model::learning_rate"),
    "bug_at_import": ("proj", "configs/config.yaml model::_target_=project.boom.TinyModel"),
    "undefined_name_in_expression": ("proj", "configs/config.yaml model::network::bias=$undefined_x"),
    "override_after_option": ("proj", "configs/config.yaml --ckpt_path last model::learnin_rate=0.01"),
    "trainer_without_target": ("proj", "configs/config.yaml ~trainer::_target_"),
    "yaml_syntax_error": ("proj", "configs/broken.yaml"),
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "__lighter__.py").touch()
    (root / "model.py").write_text(MODEL)
    (root / "boom.py").write_text("import math\n\nraise RuntimeError('boom at import')\n")
    (root / "configs" / "config.yaml").write_text(CONFIG)
    (root / "configs" / "typo.yaml").write_text(CONFIG.replace("learning_rate", "learnin_rate"))
    (root / "configs" / "broken.yaml").write_text(CONFIG.replace("  learning_rate", " learning_rate"))
    (root / "configs" / "metrics.yaml").write_text(METRICS)
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    monkeypatch.setattr(pytorch_lightning.Trainer, "fit", lambda *args, **kwargs: None)
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    yield root
    drop()


@pytest.mark.parametrize("case", CASES)
def test_mistake(case, project, monkeypatch, capsys):
    folder, args = CASES[case]
    if case == "missing_marker":
        (project / "__lighter__.py").unlink()
    monkeypatch.chdir(project.parent / folder)
    monkeypatch.setattr(sys, "argv", ["lighter", "fit", *args.split()])
    try:
        cli()
        code = 0
    except SystemExit as exit:
        code = exit.code
    out, err = capsys.readouterr()
    output = f"$ lighter fit {args}\n# exit code {code}\n{out}{err}"
    output = output.replace(str(project.parent), "<tmp>").replace(lighter.__version__, "<version>")
    output = re.sub(r"(?m)^ *[\^~]+ *\n", "", output)  # Python 3.11+ marks the failing expression, 3.10 does not
    golden = GOLDEN / f"{case}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        golden.write_text(output)
    assert output == golden.read_text()
    assert code == (1 if case.startswith("bug_") else 0 if case.startswith(("run_", "override_new")) else 2)
    assert len(err.splitlines()) <= 10 or code == 1
