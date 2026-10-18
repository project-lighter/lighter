"""Search the learning rate with Optuna (pip install optuna). Run from the project folder: python recipes/optuna_sweep.py"""

import optuna

import lighter


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 0.001, 0.3, log=True)
    model, trainer, data = lighter.build("config.yaml", {"model::optimizer::lr": lr})
    trainer.fit(model, datamodule=data)
    return trainer.callback_metrics["val/loss/epoch"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print(study.best_params)
