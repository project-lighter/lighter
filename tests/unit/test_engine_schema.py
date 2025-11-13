"""Unit tests for schema validation in lighter/engine/schema.py"""

import pytest
from sparkwheel import Config, ValidationError

from lighter.engine.schema import (
    AdapterConfig,
    AdaptersConfig,
    ArgsConfig,
    DataloadersConfig,
    LighterConfig,
    MetricsConfig,
    PredictAdapterConfig,
    SystemConfig,
)


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_empty_adapter_config(self):
        """Test creating an empty AdapterConfig."""
        config = AdapterConfig()
        assert config.batch is None
        assert config.criterion is None
        assert config.metrics is None
        assert config.logging is None

    def test_partial_adapter_config(self):
        """Test creating an AdapterConfig with some fields."""
        config = AdapterConfig(batch={"key": "value"}, metrics={"key2": "value2"})
        assert config.batch == {"key": "value"}
        assert config.criterion is None
        assert config.metrics == {"key2": "value2"}
        assert config.logging is None

    def test_full_adapter_config(self):
        """Test creating a full AdapterConfig."""
        config = AdapterConfig(
            batch={"b": 1},
            criterion={"c": 2},
            metrics={"m": 3},
            logging={"l": 4},
        )
        assert config.batch == {"b": 1}
        assert config.criterion == {"c": 2}
        assert config.metrics == {"m": 3}
        assert config.logging == {"l": 4}


class TestPredictAdapterConfig:
    """Tests for PredictAdapterConfig dataclass."""

    def test_empty_predict_adapter_config(self):
        """Test creating an empty PredictAdapterConfig."""
        config = PredictAdapterConfig()
        assert config.batch is None
        assert config.logging is None

    def test_predict_adapter_no_criterion(self):
        """Test that PredictAdapterConfig doesn't have criterion field."""
        config = PredictAdapterConfig(batch={"key": "value"})
        assert not hasattr(config, "criterion")
        assert not hasattr(config, "metrics")


class TestAdaptersConfig:
    """Tests for AdaptersConfig dataclass."""

    def test_empty_adapters_config(self):
        """Test creating an empty AdaptersConfig."""
        config = AdaptersConfig()
        assert config.train is None
        assert config.val is None
        assert config.test is None
        assert config.predict is None

    def test_partial_adapters_config(self):
        """Test creating an AdaptersConfig with some stages."""
        config = AdaptersConfig(train={"batch": {}}, val={"batch": {}})
        assert config.train is not None
        assert config.val is not None
        assert config.test is None
        assert config.predict is None


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_empty_metrics_config(self):
        """Test creating an empty MetricsConfig."""
        config = MetricsConfig()
        assert config.train is None
        assert config.val is None
        assert config.test is None

    def test_metrics_config_with_list(self):
        """Test MetricsConfig accepts lists."""
        config = MetricsConfig(train=[{"_target_": "torchmetrics.Accuracy"}])
        assert isinstance(config.train, list)
        assert len(config.train) == 1

    def test_metrics_config_with_dict(self):
        """Test MetricsConfig accepts dicts."""
        config = MetricsConfig(train={"accuracy": {"_target_": "torchmetrics.Accuracy"}})
        assert isinstance(config.train, dict)
        assert "accuracy" in config.train


class TestDataloadersConfig:
    """Tests for DataloadersConfig dataclass."""

    def test_empty_dataloaders_config(self):
        """Test creating an empty DataloadersConfig."""
        config = DataloadersConfig()
        assert config.train is None
        assert config.val is None
        assert config.test is None
        assert config.predict is None

    def test_dataloaders_config_all_stages(self):
        """Test DataloadersConfig with all stages."""
        config = DataloadersConfig(
            train={"batch_size": 32},
            val={"batch_size": 64},
            test={"batch_size": 128},
            predict={"batch_size": 256},
        )
        assert config.train["batch_size"] == 32
        assert config.val["batch_size"] == 64
        assert config.test["batch_size"] == 128
        assert config.predict["batch_size"] == 256


class TestSystemConfig:
    """Tests for SystemConfig dataclass."""

    def test_empty_system_config(self):
        """Test creating an empty SystemConfig."""
        config = SystemConfig()
        assert config.model is None
        assert config.criterion is None
        assert config.optimizer is None
        assert config.scheduler is None
        assert config.inferer is None
        assert config.metrics is None
        assert config.dataloaders is None
        assert config.adapters is None

    def test_system_config_with_nested_dataclasses(self):
        """Test SystemConfig accepts nested dataclass instances."""
        metrics = MetricsConfig(train=[])
        dataloaders = DataloadersConfig(train={})
        adapters = AdaptersConfig(train={})

        config = SystemConfig(
            model={"_target_": "torch.nn.Identity"},
            metrics=metrics,
            dataloaders=dataloaders,
            adapters=adapters,
        )
        assert config.model is not None
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.dataloaders, DataloadersConfig)
        assert isinstance(config.adapters, AdaptersConfig)


class TestArgsConfig:
    """Tests for ArgsConfig dataclass."""

    def test_empty_args_config(self):
        """Test creating an empty ArgsConfig."""
        config = ArgsConfig()
        assert config.fit is None
        assert config.validate is None
        assert config.test is None
        assert config.predict is None

    def test_args_config_all_stages(self):
        """Test ArgsConfig with all stages."""
        config = ArgsConfig(
            fit={"ckpt_path": "checkpoint.ckpt"},
            validate={"verbose": True},
            test={"verbose": False},
            predict={"return_predictions": True},
        )
        assert config.fit["ckpt_path"] == "checkpoint.ckpt"
        assert config.validate["verbose"] is True
        assert config.test["verbose"] is False
        assert config.predict["return_predictions"] is True


class TestLighterConfig:
    """Tests for main LighterConfig schema."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration with only required fields."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("trainer::max_epochs") == 10
        assert config.get("system::model::_target_") == "torch.nn.Identity"

    def test_full_valid_config(self):
        """Test full valid configuration with all optional fields."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss"},
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
                "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 10},
                "inferer": {"_target_": "lighter.Inferer"},
                "metrics": {"train": [], "val": [], "test": []},
                "dataloaders": {"train": {}, "val": {}, "test": {}, "predict": {}},
                "adapters": {"train": {}, "val": {}, "test": {}, "predict": {}},
            },
            "project": "./path/to/project",
            "vars": {"learning_rate": 0.001},
            "args": {"fit": {}, "validate": {}, "test": {}, "predict": {}},
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("project") == "./path/to/project"
        assert config.get("vars::learning_rate") == 0.001
        assert config.get("system::optimizer::lr") == 0.001

    def test_missing_trainer_raises_error(self):
        """Test that missing trainer field raises ValidationError."""
        config_dict = {
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            Config.load(config_dict, schema=LighterConfig)
        assert "trainer" in str(exc_info.value).lower()

    def test_missing_system_raises_error(self):
        """Test that missing system field raises ValidationError."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
        }
        with pytest.raises(ValidationError) as exc_info:
            Config.load(config_dict, schema=LighterConfig)
        assert "system" in str(exc_info.value).lower()

    def test_trainer_wrong_type_raises_error(self):
        """Test that trainer with wrong type raises ValidationError."""
        config_dict = {
            "trainer": ["not", "a", "dict"],  # Should be dict, not list
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
        }
        with pytest.raises(ValidationError):
            Config.load(config_dict, schema=LighterConfig)

    def test_system_wrong_type_raises_error(self):
        """Test that system with wrong type raises ValidationError."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": "not a dict",  # Should be SystemConfig/dict
        }
        with pytest.raises(ValidationError):
            Config.load(config_dict, schema=LighterConfig)

    def test_optional_project_field(self):
        """Test that project field is optional."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
            "project": "./my_project",
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("project") == "./my_project"

    def test_optional_vars_field(self):
        """Test that vars field is optional and accepts any dict."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
            "vars": {"custom_var": 42, "another_var": "value"},
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("vars::custom_var") == 42
        assert config.get("vars::another_var") == "value"

    def test_nested_metrics_structure(self):
        """Test nested metrics configuration structure."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "metrics": {
                    "train": [{"_target_": "torchmetrics.Accuracy", "task": "binary"}],
                    "val": [{"_target_": "torchmetrics.F1Score", "task": "binary"}],
                },
            },
        }
        config = Config.load(config_dict, schema=LighterConfig)
        train_metrics = config.get("system::metrics::train")
        assert isinstance(train_metrics, list)
        assert len(train_metrics) == 1
        assert train_metrics[0]["_target_"] == "torchmetrics.Accuracy"

    def test_nested_dataloaders_structure(self):
        """Test nested dataloaders configuration structure."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "dataloaders": {
                    "train": {"_target_": "torch.utils.data.DataLoader", "batch_size": 32},
                    "val": {"_target_": "torch.utils.data.DataLoader", "batch_size": 64},
                },
            },
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("system::dataloaders::train::batch_size") == 32
        assert config.get("system::dataloaders::val::batch_size") == 64

    def test_nested_adapters_structure(self):
        """Test nested adapters configuration structure."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "adapters": {
                    "train": {
                        "batch": {"_target_": "lighter.adapters.BatchAdapter", "input_accessor": 0},
                    },
                },
            },
        }
        config = Config.load(config_dict, schema=LighterConfig)
        batch_adapter = config.get("system::adapters::train::batch")
        assert batch_adapter["_target_"] == "lighter.adapters.BatchAdapter"
        assert batch_adapter["input_accessor"] == 0

    def test_args_for_all_stages(self):
        """Test args configuration for all stages."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
            "args": {
                "fit": {"ckpt_path": "checkpoint.ckpt"},
                "validate": {"verbose": True},
                "test": {"verbose": False},
                "predict": {"return_predictions": True},
            },
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("args::fit::ckpt_path") == "checkpoint.ckpt"
        assert config.get("args::validate::verbose") is True
        assert config.get("args::test::verbose") is False
        assert config.get("args::predict::return_predictions") is True

    def test_extra_fields_rejected(self):
        """Test that extra fields not in schema are rejected (Sparkwheel behavior)."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {"_target_": "lighter.System", "model": {"_target_": "torch.nn.Identity"}},
            "extra_field": "this is extra",
        }
        # Sparkwheel validates strictly against the schema
        with pytest.raises(ValidationError, match="extra_field"):
            Config.load(config_dict, schema=LighterConfig)

    def test_config_with_references(self):
        """Test configuration with Sparkwheel references (@, %, $)."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "metrics": {
                    "train": [{"_target_": "torchmetrics.Accuracy", "task": "binary"}],
                    "val": "%::train",  # Reference to train metrics
                },
            },
        }
        config = Config.load(config_dict, schema=LighterConfig)
        # Before resolution, this is a reference string
        assert config.get("system::metrics::val") == "%::train"

    def test_empty_nested_configs(self):
        """Test empty nested configurations are valid."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "metrics": {},
                "dataloaders": {},
                "adapters": {},
            },
        }
        config = Config.load(config_dict, schema=LighterConfig)
        assert config.get("system::metrics") == {}
        assert config.get("system::dataloaders") == {}
        assert config.get("system::adapters") == {}
