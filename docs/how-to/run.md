## Stages

Lighter uses PyTorch Lightning's Trainer to manage deep learning experiments. The available stages are:

*   **`fit`**: Train your model on training data.
*   **`validate`**: Evaluate model performance on validation data
*   **`test`**: Evaluate final model performance on test data
*   **`predict`**: Generate predictions on new data
*   **`lr_find`**: Find optimal learning rate
*   **`scale_batch_size`**: Find largest batch size that fits in GPU memory

For documentation on each, please refer to PyTorch Lightning: [`fit`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit), [`validate`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#validate), [`test`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#test), [`predict`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#predict), [`lr_find`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.lr_find), [`scale_batch_size`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.scale_batch_size).

## Running a Stage

The basic command to run a stage of an experiment is:

```bash
lighter <stage> config.yaml
```

Where:

*   `<stage>` is one of the stages mentioned above (e.g., `fit`, `validate`, `test`, `predict`, `lr_find`, `scale_batch_size`).
*   `config.yaml` is the configuration file that defines your experiment. You can also define multiple config files separated by commas, which will be merged (e.g., `config1.yaml,config2.yaml`).

For example, to train your model, you would use:

```bash
lighter fit config.yaml
```

### Passing arguments to a stage

To pass arguments to a stage, use the `args` section in in your config. For example, to set the `ckpt_path` argument of the `fit` stage/method in your config:

```yaml
args:
    fit:
        ckpt_path: "path/to/checkpoint.ckpt"
```

or pass/override it from the command line:

```bash
lighter fit experiment.yaml --args#fit#ckpt_path="path/to/checkpoint.ckpt"
```

## Recap and Next Steps

You now know how to run different stages of your experiment using Lighter. Next, explore the [Project Module](project_module.md) to learn how to organize your project and reference custom modules in your configuration.