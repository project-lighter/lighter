# Running Workflows

Once the configuration is established, Lighter can run different deep learning workflows. The following workflows are supported:

1. fit
2. validate
3. test
4. predict

These workflows are inherited from the PyTorch Lightning trainer and can be found in the [PL docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#methods).

Below, we show how you can run these workflows and what are some "required" definitions in the config while running these workflows.

## Fit workflow
The fit workflow is used for training the model. It requires the `trainer` and `system` configurations to be defined in the YAML file.

## Validate workflow
The validate workflow is used to evaluate the model on a validation dataset. Ensure that the `val` dataset is defined in the `system` configuration.

## Test workflow
The test workflow is used to evaluate the model on a test dataset. Ensure that the `test` dataset is defined in the `system` configuration.

## Predict workflow
The predict workflow is used to make predictions on new data. Ensure that the `predict` dataset is defined in the `system` configuration.

## Fit workflow


## Validate workflow

## Test workflow

## Predict workflow
