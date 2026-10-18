"""Unit tests for error handling in the Runner class"""

import pytest
from sparkwheel.utils.exceptions import ConfigKeyError

from lighter.engine.runner import Runner, build, load_config


class TestRunnerErrorHandling:
    """Test class for Runner error handling scenarios."""

    def test_run_without_config_raises_error(self):
        """Test that calling run without config_paths raises ConfigKeyError."""
        with pytest.raises(ConfigKeyError):  # Sparkwheel raises ConfigKeyError for missing keys
            Runner().run("fit", [])

    def test_run_with_nonexistent_config_raises_error(self):
        """Test that nonexistent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Runner().run("fit", ["/nonexistent/path/config.yaml"])

    def test_run_with_empty_config_raises_validation_error(self, tmp_path):
        """Test that empty config raises ConfigKeyError when resolving 'model'."""
        (tmp_path / "empty.yaml").write_text("")
        with pytest.raises(ConfigKeyError):
            Runner().run("fit", [str(tmp_path / "empty.yaml")])

    def test_multiple_config_files(self, tmp_path):
        """Several config files are merged in order."""
        (tmp_path / "a.yaml").write_text("trainer:\n  _target_: pytorch_lightning.Trainer\n  max_epochs: 1\n")
        (tmp_path / "b.yaml").write_text(
            "model:\n  _target_: lighter.LighterModule\n  network:\n    _target_: torch.nn.Identity\n"
        )
        config = load_config(str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"))
        assert config.get("trainer::max_epochs") == 1
        assert config.get("model::network::_target_") == "torch.nn.Identity"

    def test_data_must_be_a_datamodule(self):
        """A data key that builds something else raises TypeError."""
        with pytest.raises(TypeError, match="`data` must build a LightningDataModule, got dict"):
            build(
                {
                    "model": {"_target_": "fixtures.plain_lightning_modules.PlainLightningModule"},
                    "trainer": {"_target_": "pytorch_lightning.Trainer", "logger": False},
                    "data": {"_target_": "builtins.dict"},
                }
            )
