Lighter uses simple **YAML config files** to define your experiments. Think of a config as a recipe for your experiments, telling Lighter what to train, how to train it, and with what data.  This keeps your experiment setup clean and easy to understand.

Every Lighter config needs **two main parts**: `trainer` and `system`.

## Core Sections: `trainer` and `system`

*   **`trainer`**: This section is all about [`pytorch_lightning.Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html). You configure *how* your model will be trained here, like how many epochs to run.

*   **`system`**: This is where you define *what* you are training. Using [`lighter.System`](../../reference/system/#lighter.system.System), you specify the key ingredients of your experiment:

    *   **`model`**: Your neural network itself (e.g., ResNet, Linear layers).
    *   **`criterion`**: The loss function (e.g., CrossEntropyLoss).
    *   **`optimizer`**: How the model learns (e.g., Adam).
    *   **`dataloaders`**: How your data is loaded for training and validation.

Let's look at a super simple example:

```yaml title="config.yaml"
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 5  # Train for 5 epochs

system:
    _target_: lighter.System

    model:
        _target_: torch.nn.Linear  # A simple linear model
        in_features: 784
        out_features: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()" # Link to model's learnable parameters
        lr: 0.001

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            dataset:
                _target_: ... # (Dataset definition will come later in tutorials)
            batch_size: 64
```

## Key Config Concepts

### `_target_`:  Saying "What to Use"

The special key `_target_` is how you tell Lighter *what* Python class or function to use.  For example:

```yaml hl_lines="5"
system:
    _target_: lighter.System

    model:
        _target_: torch.nn.Linear  # Use the Linear layer from torch.nn
        in_features: 784
        out_features: 10
```

`_target_: torch.nn.Linear` means "use the `Linear` class from the `torch.nn` module".  Any other settings under `model` (like `in_features` and `out_features`) become the *arguments* used when creating this `Linear` layer.

### `@` and `%` Referencing: Connecting Things Together

Sometimes you need to link parts of your config together. We use `@` or `%` for this. The main difference is that `@` references the *instance* of an object, while `%` references the *definition* of an object. We'll dive deeper into the difference between the two at a later stage. For the moment, we will only use `@`.

A common example is telling the optimizer *which parameters* to update – these are the parameters of your `model`:

```yaml hl_lines="6"
system:
    # ... model definition ...

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001
```

`"@system#model.parameters()"` means:

1.  `@system#model`:  "Go to the `system` section and find the `model` we defined."
2.  `.parameters()`: "Then, get the `parameters()` method of that model."
3.  `$`: "Evaluate this as a Python expression."

So, `params: "$@system#model.parameters()"` nicely links the `Adam` optimizer to the learnable weights of your `Linear` model.

In Python, this is equivalent to:

```python
model = torch.nn.Linear(in_features=784, out_features=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Command Line Overrides

You can easily change settings in your config directly from the command line when you run your experiment. For instance, to train for 10 epochs instead of 5 (defined in `config.yaml`):

```bash
lighter fit config.yaml --trainer#max_epochs=10
```

This is super handy for quick experiments!

## That's the Basics!

For these tutorials, you'll mainly use `trainer` and `system` sections, `_target_` to define components, and `@`. **For thorough explanation of the config system, please refer to [Configure](../how-to/configure.md) How-To guide**.

In the next tutorial, we'll use these config basics to build an [image classification](image_classification.md) experiment!
