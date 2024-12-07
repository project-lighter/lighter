# Postprocessing

Postprocessing in Lighter allows you to apply custom transformations to data at various stages of the workflow. This can include modifying inputs, targets, predictions, or entire batches.

## Defining Postprocessing Functions
Postprocessing functions can be defined in the configuration file under the `postprocessing` key. They can be applied to:
- **Batch**: Modify the entire batch before it is passed to the model.
- **Criterion**: Modify inputs, targets, or predictions before loss calculation.
- **Metrics**: Modify inputs, targets, or predictions before metric calculation.
- **Logging**: Modify inputs, targets, or predictions before logging.

## Example
```yaml
postprocessing:
  batch:
    train: '$lambda x: {"input": x[0], "target": x[1]}'
  criterion:
    input: '$lambda x: x / 255.0'
  metrics:
    pred: '$lambda x: x.argmax(dim=1)'
  logging:
    target: '$lambda x: x.cpu().numpy()'
```

For more information on how to use postprocessing in Lighter, refer to the [Lighter documentation](./config.md).
