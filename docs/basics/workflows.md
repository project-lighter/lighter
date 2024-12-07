# Running Workflows

With your configuration in place, Lighter can execute various deep learning workflows. The supported workflows include:

1. fit
2. validate
3. test
4. predict

These workflows are inherited from the PyTorch Lightning trainer and can be found in the [PL docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#methods).

Below, we outline how to run these workflows and highlight the essential configuration elements required for each.

## Fit workflow
The fit workflow is designed for training models. Ensure that both the `trainer` and `system` configurations are specified in your YAML file.

## Validate workflow
The validate workflow evaluates models on a validation dataset. Make sure the `val` dataset is defined within the `system` configuration.

## Test workflow
The test workflow assesses models using a test dataset. Confirm that the `test` dataset is included in the `system` configuration.

## Predict workflow
The predict workflow generates predictions on new data. Verify that the `predict` dataset is specified in the `system` configuration.
