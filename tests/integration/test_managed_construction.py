"""Native optimizer setup with ordinary configuration and scientific step methods."""

import copy
import pickle

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from sparkwheel import Config, InstantiationError
from sparkwheel.construction import BlockedPathError
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lighter import LighterModule, Runner
from lighter.engine.runner import _ResolutionView
from lighter.utils.types.enums import Stage


class Task(LighterModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.criterion(self.network(x), y)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


class Capture(Callback):
    runs = []

    def __init__(self, model):
        self.model = model
        self.initial_optimizer = model.optimizer
        self.initial_scheduler = model.scheduler
        self.initial_parameters = [p.detach().clone() for p in model.parameters()]
        self.trainer = None
        self.runs.append(self)

    def on_fit_start(self, trainer, pl_module):
        assert pl_module is self.model
        self.trainer = trainer

    def on_test_start(self, trainer, pl_module):
        assert pl_module is self.model
        self.trainer = trainer


def loader():
    return DataLoader(TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[2.0], [4.0], [6.0]])), batch_size=2)


def trainer_options(tmp_path):
    return {
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": str(tmp_path),
    }


def recipe(tmp_path, task=Task):
    return {
        "seed": 17,
        "model": {
            "_target_": f"{__name__}.{task.__name__}",
            "network": {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
            "criterion": {"_target_": "torch.nn.MSELoss"},
            "optimizer": {"_target_": "torch.optim.SGD", "params": "$@model::network.parameters()", "lr": 0.05},
            "scheduler": {
                "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "optimizer": "@model::optimizer", "step_size": 1},
                "interval": "epoch",
                "frequency": 1,
            },
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            **trainer_options(tmp_path),
            "callbacks": [{"_target_": f"{__name__}.Capture", "model": "@model"}],
        },
        "args": {"fit": {"train_dataloaders": {"_target_": f"{__name__}.loader"}}},
    }


def run(tmp_path, config=None, **kwargs):
    Runner().run("fit", [recipe(tmp_path) if config is None else config], **kwargs)
    return Capture.runs[-1]


class NativeTask(LightningModule):
    def __init__(self, weights):
        super().__init__()
        self.network = nn.Linear(1, 1)
        self.network.load_state_dict(weights)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self.network(x), y)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), lr=0.05)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 1)}}


def test_step_only_recipe_constructs_at_native_setup_with_native_updates(tmp_path):
    config = recipe(tmp_path)
    original = copy.deepcopy(config)
    captured = run(tmp_path, config)
    model = captured.model
    assert captured.initial_optimizer is None
    assert captured.initial_scheduler is None
    assert isinstance(model.optimizer, torch.optim.SGD)
    assert model.scheduler["scheduler"].optimizer is model.optimizer
    assert model.optimizer is captured.trainer.optimizers[0]
    weights = dict(zip(model.network.state_dict(), captured.initial_parameters, strict=True))
    native = NativeTask(weights)
    Trainer(**trainer_options(tmp_path)).fit(native, loader())
    for actual, expected in zip(model.network.parameters(), native.network.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert config == original
    assert model.hparams["config"]["model"]["optimizer"]["params"] == "$@model::network.parameters()"


def test_repeated_native_setup_refreshes_iterators_optimizers_and_schedulers(tmp_path):
    captured = run(tmp_path)
    model = captured.model
    previous = model.optimizer
    for _ in range(3):
        configured = model.configure_optimizers()
        assert configured["optimizer"] is model.optimizer is not previous
        assert configured["lr_scheduler"]["scheduler"].optimizer is model.optimizer
        assert [id(p) for group in model.optimizer.param_groups for p in group["params"]] == [
            id(p) for p in model.network.parameters()
        ]
        previous = model.optimizer
    assert not hasattr(model._optimizer_binding, "scope")
    restored = pickle.loads(pickle.dumps(model))
    restored.configure_optimizers()
    assert restored.optimizer.param_groups[0]["params"][0] is restored.network.weight


def test_independent_runs_reconstruct_objects_with_same_seed(tmp_path):
    config = recipe(tmp_path)
    first = run(tmp_path, config)
    second = run(tmp_path, config)
    assert first.model is not second.model
    assert first.model.network is not second.model.network
    assert first.model.optimizer is not second.model.optimizer
    for a, b in zip(first.model.parameters(), second.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fit_then_native_test_preserves_network_and_weights(tmp_path):
    captured = run(tmp_path)
    network = captured.model.network
    weights = {key: value.clone() for key, value in network.state_dict().items()}
    captured.trainer.test(captured.model, loader())
    assert captured.model.network is network
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_native_checkpoint_resume_restores_optimizer_state_after_fresh_setup(tmp_path):
    first = run(tmp_path)
    checkpoint = tmp_path / "continue.ckpt"
    first.trainer.save_checkpoint(checkpoint)
    config = recipe(tmp_path)
    config["trainer"]["max_epochs"] = 2
    config["model"]["optimizer"]["lr"] = 0.9
    second = run(tmp_path, config, ckpt_path=str(checkpoint))
    assert second.trainer.global_step == first.trainer.global_step * 2
    assert second.model.optimizer is not first.model.optimizer
    # StepLR restored its epoch and the optimizer's LR, then completed epoch 2.
    assert second.model.optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)


def test_copy_alias_model_and_native_parameter_groups_preserve_membership_and_order(tmp_path):
    config = recipe(tmp_path)
    config["model"]["network"] = {
        "_target_": "torch.nn.Sequential",
        "_args_": [
            {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
            {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
        ],
    }
    config["model"]["optimizer"]["params"] = [
        {"params": "$@model::network[1].parameters()", "lr": 0.01, "weight_decay": 0.2},
        {"params": "$[@model::network[0].weight]", "lr": 0.02},
    ]
    config["template"] = config.pop("model")
    config["model"] = "%template"
    captured = run(tmp_path, config)
    groups = captured.model.optimizer.param_groups
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in captured.model.network[1].parameters()]
    assert groups[0]["weight_decay"] == 0.2
    assert groups[1]["params"] == [captured.model.network[0].weight]
    assert groups[0]["initial_lr"] == 0.01
    assert groups[1]["initial_lr"] == 0.02
    assert all(captured.model.network[0].bias is not p for group in groups for p in group["params"])
    assert config["model"] == "%template"


class ReplaceParameter(Task):
    def configure_model(self):
        self.network.weight = nn.Parameter(self.network.weight.detach().clone())


class ReplaceNetwork(Task):
    def configure_model(self):
        self.network = nn.Linear(1, 1)


class ReplaceSharedChild(Task):
    def configure_model(self):
        self.network[0] = nn.Linear(1, 1)


def test_parameter_replacement_within_network_uses_live_parameters(tmp_path):
    captured = run(tmp_path, recipe(tmp_path, ReplaceParameter))
    assert captured.model.optimizer.param_groups[0]["params"][0] is captured.model.network.weight


def test_whole_network_replacement_reports_conflicting_source_field(tmp_path):
    with pytest.raises(ValueError, match="model::network.*shared aliases"):
        run(tmp_path, recipe(tmp_path, ReplaceNetwork))


def test_replaced_shared_child_reports_old_alias_instead_of_optimizing_old_parameters(tmp_path):
    config = recipe(tmp_path, ReplaceSharedChild)
    config["shared"] = config["model"]["network"]
    config["model"]["network"] = {"_target_": "torch.nn.Sequential", "_args_": ["@shared"]}
    config["model"]["optimizer"]["params"] = "$@shared.parameters()"
    with pytest.raises(ValueError, match="alias 'shared'.*live model::network.0"):
        run(tmp_path, config)


def test_early_optimizer_dependency_fails_with_source_path(tmp_path):
    config = recipe(tmp_path)
    config["trainer"]["callbacks"][0]["model"] = "@model::optimizer"
    with pytest.raises(ValueError, match="model::optimizer.*blocked|blocked.*model::optimizer"):
        run(tmp_path, config)


class CustomConstructor(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.optimizer is not None


class CustomHook(Task):
    calls = 0

    def configure_optimizers(self):
        type(self).calls += 1
        assert getattr(self, "_optimizer_binding", None) is None
        return super().configure_optimizers()


@pytest.mark.parametrize("task", [CustomConstructor, CustomHook])
def test_custom_constructor_or_optimizer_hook_retains_eager_ownership(tmp_path, task):
    captured = run(tmp_path, recipe(tmp_path, task))
    assert captured.initial_optimizer is captured.model.optimizer
    assert getattr(captured.model, "_optimizer_binding", None) is None


def test_prebuilt_optimizer_and_prebuilt_module_preserve_identity(tmp_path):
    network = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
    config = recipe(tmp_path)
    config["model"]["network"] = network
    config["model"]["optimizer"] = optimizer
    config["model"].pop("scheduler")
    captured = run(tmp_path, config)
    assert captured.model.optimizer is optimizer
    assert getattr(captured.model, "_optimizer_binding", None) is None
    config["model"] = captured.model
    second = run(tmp_path, config)
    assert second.model is captured.model
    assert second.model.optimizer is optimizer


class ManualTask(Task):
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = super().training_step(batch, batch_idx)
        self.manual_backward(loss)
        optimizer.step()
        return loss


def test_inherited_manual_optimization_remains_manual(tmp_path):
    captured = run(tmp_path, recipe(tmp_path, ManualTask))
    assert captured.model.automatic_optimization is False
    assert captured.trainer.global_step == 2


def fail_if_constructed():
    raise AssertionError("Inactive or overridden stage recipe constructed")


def test_managed_scope_keeps_authoritative_overrides_and_inactive_blocks(tmp_path):
    config = recipe(tmp_path)
    config["args"]["fit"]["train_dataloaders"] = {
        "_target_": f"{__name__}.fail_if_constructed",
        "side_effect": "$import no_such_module",
    }
    config["args"]["test"] = {"dataloaders": {"_target_": f"{__name__}.fail_if_constructed"}}
    config["model"]["optimizer"]["lr"] = "$0.05 if @args::fit::train_dataloaders is not None else 0.0"
    captured = run(tmp_path, config, train_dataloaders=loader())
    assert captured.model.optimizer.param_groups[0]["initial_lr"] == 0.05


def test_retained_source_has_no_constructed_components(tmp_path):
    config = Config(recipe(tmp_path))
    source_path = tmp_path / "source.yaml"
    Config.export_config_file(config.get(), source_path)
    before = source_path.read_text()
    Runner().run("fit", [str(source_path)])
    Config.export_config_file(config.get(), source_path)
    assert source_path.read_text() == before
    assert "parameters()" in before


def test_scheduler_expressions_bind_the_actual_model_and_optimizer(tmp_path):
    config = recipe(tmp_path)
    config["model"]["optimizer"]["params"] = "$@model.network.parameters()"
    config["model"]["scheduler"]["scheduler"]["optimizer"] = "$@model.optimizer"
    captured = run(tmp_path, config)
    assert captured.model.scheduler["scheduler"].optimizer is captured.model.optimizer


def test_preserved_shared_network_alias_remains_one_live_object(tmp_path):
    config = recipe(tmp_path)
    config["shared"] = config["model"]["network"]
    config["model"]["network"] = "@shared"
    config["model"]["optimizer"]["params"] = "$@shared.parameters()"
    captured = run(tmp_path, config)
    assert captured.model._optimizer_binding.components["shared"] is captured.model.network
    captured.model.configure_optimizers()
    assert captured.model.optimizer.param_groups[0]["params"][0] is captured.model.network.weight


class ParameterNetwork(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


def test_replaced_explicit_parameter_alias_is_rejected(tmp_path):
    config = recipe(tmp_path, ReplaceParameter)
    config["weight"] = {"_target_": "torch.nn.Parameter", "data": {"_target_": "torch.ones", "_args_": [1, 1]}}
    config["model"]["network"] = {"_target_": f"{__name__}.ParameterNetwork", "weight": "@weight"}
    config["model"]["optimizer"]["params"] = "$[@weight]"
    with pytest.raises(ValueError, match="alias 'weight'.*live model::network.weight"):
        run(tmp_path, config)


def test_optimizer_dependent_eager_field_fails_before_constructor(tmp_path):
    config = recipe(tmp_path)
    config["model"]["criterion"] = "@model::optimizer"
    with pytest.raises(ValueError, match="model::optimizer.*blocked|blocked.*model::optimizer"):
        run(tmp_path, config)


def test_prebuilt_parameter_iterator_retains_eager_ownership(tmp_path):
    config = recipe(tmp_path)
    network = nn.Linear(1, 1)
    config["model"]["network"] = network
    config["model"]["optimizer"]["params"] = network.parameters()
    model = Runner()._resolve_model(_ResolutionView(Config(config), Stage.FIT, {}))
    assert getattr(model, "_optimizer_binding", None) is None
    assert model.optimizer.param_groups[0]["params"][0] is network.weight


def test_managed_scope_rejects_inactive_stage_reference_at_optimizer_setup(tmp_path):
    config = recipe(tmp_path)
    config["args"]["test"] = {"dataloaders": {"_target_": f"{__name__}.fail_if_constructed"}}
    config["model"]["optimizer"]["lr"] = "$0.05 if @args::test::dataloaders is not None else 0.0"
    with pytest.raises(ValueError, match="args::test.*blocked|blocked.*args::test"):
        run(tmp_path, config)


def test_alias_to_prebuilt_optimizer_preserves_native_identity(tmp_path):
    network = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
    config = recipe(tmp_path)
    config["model"]["network"] = network
    config["existing_optimizer"] = optimizer
    config["model"]["optimizer"] = "@existing_optimizer"
    captured = run(tmp_path, config)
    assert captured.initial_optimizer is optimizer is captured.model.optimizer
    assert getattr(captured.model, "_optimizer_binding", None) is None


def test_alias_to_prebuilt_iterator_retains_native_ownership(tmp_path):
    network = nn.Linear(1, 1)
    config = recipe(tmp_path)
    config["model"]["network"] = network
    config["params_iterator"] = network.parameters()
    config["model"]["optimizer"]["params"] = "@params_iterator"
    model = Runner()._resolve_model(_ResolutionView(Config(config), Stage.FIT, {}))
    assert getattr(model, "_optimizer_binding", None) is None
    assert model.optimizer.param_groups[0]["params"][0] is network.weight


def test_prebuilt_network_alone_still_uses_managed_optimizer_setup(tmp_path):
    config = recipe(tmp_path)
    network = nn.Linear(1, 1)
    config["model"]["network"] = network
    captured = run(tmp_path, config)
    assert captured.initial_optimizer is None
    assert captured.model.network is network
    captured.model.configure_optimizers()
    assert captured.model.optimizer.param_groups[0]["params"][0] is network.weight


def test_reusing_prebuilt_managed_model_keeps_original_binding_ownership(tmp_path):
    config = recipe(tmp_path)
    config["shared"] = config["model"]["network"]
    config["model"]["network"] = "@shared"
    config["model"]["optimizer"]["params"] = "$@shared.parameters()"
    first = run(tmp_path, config)
    components = first.model._optimizer_binding.components
    second = recipe(tmp_path)
    second["model"] = first.model
    second["args"] = {"test": {"dataloaders": {"_target_": f"{__name__}.loader"}}}
    Runner().run("test", [second])
    assert first.model._optimizer_binding.components is components
    first.model.configure_optimizers()
    assert first.model.optimizer.param_groups[0]["params"][0] is first.model.network.weight


def test_managed_plateau_dictionary_uses_native_monitor_and_scheduler_identity(tmp_path):
    config = recipe(tmp_path)
    config["model"]["scheduler"] = {
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "optimizer": "@model::optimizer",
            "patience": 0,
        },
        "monitor": "train/loss/epoch",
        "interval": "epoch",
        "frequency": 1,
        "strict": True,
    }
    captured = run(tmp_path, config)
    actual = captured.trainer.lr_scheduler_configs[0]
    assert actual.scheduler.optimizer is captured.model.optimizer
    assert actual.monitor == "train/loss/epoch"
    assert actual.strict is True
    assert actual.reduce_on_plateau is True
    assert actual.scheduler.best < float("inf")


def test_evaluation_only_does_not_construct_unneeded_optimizer(tmp_path):
    config = recipe(tmp_path)
    config["model"]["optimizer"] = {"_target_": f"{__name__}.fail_if_constructed"}
    config["args"] = {"test": {"dataloaders": {"_target_": f"{__name__}.loader"}}}
    Runner().run("test", [config])
    captured = Capture.runs[-1]
    assert captured.model.optimizer is None
    assert captured.model.scheduler is None
    assert captured.trainer.callback_metrics["test/loss/epoch"].isfinite()


class DiagnosticValue(Callback):
    def __init__(self, value):
        self.value = value


@pytest.fixture
def managed_diagnostic_construction(tmp_path, monkeypatch):
    events = []
    for name, cls in (
        ("network", nn.Linear),
        ("model", LighterModule),
        ("trainer", Trainer),
        ("callback", DiagnosticValue),
        ("optimizer", torch.optim.SGD),
    ):
        original = cls.__init__

        def counted(self, *args, _name=name, _original=original, **kwargs):
            events.append(_name)
            _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", counted)
    config = recipe(tmp_path)
    config["trainer"]["callbacks"] = [{"_target_": f"{__name__}.DiagnosticValue", "value": "@learning_rate"}]
    config["learning_rate"] = 0.05
    return config, events


@pytest.mark.parametrize(
    ("dependency", "requested", "blocked"),
    [
        ("@model::optimizer", "model::optimizer", "model::optimizer"),
        ("@model::optimizer::lr", "model::optimizer::lr", "model::optimizer"),
        ("$@model::optimizer::lr + 0", "model::optimizer::lr", "model::optimizer"),
        ("@model::scheduler::scheduler::step_size", "model::scheduler::scheduler::step_size", "model::scheduler"),
    ],
)
def test_managed_diagnostic_callback_paths_preserve_cause_before_construction(
    managed_diagnostic_construction, dependency, requested, blocked
):
    config, events = managed_diagnostic_construction
    config["trainer"]["callbacks"][0]["value"] = dependency
    view = _ResolutionView(Config(data=config), Stage.FIT, {})
    Runner()._resolve_model(view)
    with pytest.raises(ValueError, match="Runner-managed.*configure_optimizers") as caught:
        Runner()._resolve_trainer(view)
    cause = caught.value.__cause__
    assert isinstance(cause, BlockedPathError)
    assert cause.requested_path == requested
    assert cause.blocked_path == blocked
    assert str(caught.value).startswith(str(cause) + "\n")
    assert "top-level scalar" in str(caught.value)
    assert "ordinary Trainer.fit" in str(caught.value)
    assert "trainer.optimizers in on_train_start" in str(caught.value)
    assert "custom lifecycle owns availability" in str(caught.value)
    assert events == ["network", "model"]


def test_managed_diagnostic_eager_field_uses_local_deferred_paths(managed_diagnostic_construction):
    config, events = managed_diagnostic_construction
    config["model"]["criterion"] = "@model::optimizer"
    view = _ResolutionView(Config(data=config), Stage.FIT, {})
    with pytest.raises(ValueError, match="Runner-managed.*configure_optimizers") as caught:
        Runner()._resolve_model(view)
    assert view.managed_binding is None
    cause = caught.value.__cause__
    assert isinstance(cause, BlockedPathError)
    assert cause.requested_path == cause.blocked_path == "model::optimizer"
    assert events == ["network"]


def test_managed_diagnostic_shared_requested_scalar_requires_no_optimizer(managed_diagnostic_construction):
    config, events = managed_diagnostic_construction
    view = _ResolutionView(Config(data=config), Stage.FIT, {})
    model = Runner()._resolve_model(view)
    trainer = Runner()._resolve_trainer(view)
    callback = next(callback for callback in trainer.callbacks if isinstance(callback, DiagnosticValue))
    assert callback.value == 0.05
    assert model.optimizer is model.scheduler is None
    assert events == ["network", "model", "callback", "trainer"]


def test_managed_diagnostic_inactive_stage_blocker_has_no_optimizer_guidance(managed_diagnostic_construction):
    config, events = managed_diagnostic_construction
    config["args"]["test"] = {"dataloaders": {"_target_": f"{__name__}.fail_if_constructed"}}
    config["trainer"]["callbacks"][0]["value"] = "@args::test::dataloaders"
    view = _ResolutionView(Config(data=config), Stage.FIT, {})
    Runner()._resolve_model(view)
    with pytest.raises(BlockedPathError) as caught:
        Runner()._resolve_trainer(view)
    assert caught.value.requested_path == "args::test::dataloaders"
    assert caught.value.blocked_path == "args::test"
    assert caught.value.__cause__ is None
    assert "Runner-managed" not in str(caught.value)
    assert events == ["network", "model"]


def test_managed_diagnostic_does_not_scrape_unrelated_constructor_errors(managed_diagnostic_construction):
    config, events = managed_diagnostic_construction
    original = ValueError("Cannot resolve 'model::optimizer::lr': construction path 'model::optimizer' is blocked")

    def fail():
        raise original

    config["trainer"]["callbacks"] = [{"_target_": fail}]
    view = _ResolutionView(Config(data=config), Stage.FIT, {})
    Runner()._resolve_model(view)
    with pytest.raises(InstantiationError) as caught:
        Runner()._resolve_trainer(view)
    assert str(original) in str(caught.value)
    chain = []
    error = caught.value
    while error is not None:
        chain.append(error)
        assert "Runner-managed" not in str(error)
        error = error.__cause__
    assert original in chain
    assert events == ["network", "model"]
