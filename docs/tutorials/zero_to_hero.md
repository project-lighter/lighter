# Zero to Hero: Your First Lighter Experiment

Welcome! In the next 15 minutes, you'll go from never having used Lighter to running your own deep learning experiments with just YAML configs. No boilerplate code required.

## What You'll Achieve

By the end of this tutorial, you'll:

- ✅ Understand the core philosophy of config-driven ML
- ✅ Run your first experiment with a single command
- ✅ Know how to customize any part of your training
- ✅ Feel confident exploring the rest of Lighter's features

Let's get started!

## The Big Idea

Traditional PyTorch Lightning code looks like this:

```python
# 50+ lines of boilerplate...
class MyModel(LightningModule):
    def __init__(self):
        self.model = resnet18()
        self.criterion = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

# ... more boilerplate ...

trainer = Trainer(max_epochs=10)
trainer.fit(model, train_loader)
```

**With Lighter, that becomes:**

```yaml
# config.yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: torchvision.models.resnet18
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001
  dataloaders:
    train: ...
```

```bash
lighter fit config.yaml
```

**Same functionality. Zero boilerplate. Pure configuration.**

## Your First Experiment (5 Minutes)

Let's run a real experiment. Create a file called `quick_start.yaml`:

```yaml title="quick_start.yaml"
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 3

system:
  _target_: lighter.System

  model:
    _target_: torch.nn.Linear
    in_features: 784  # MNIST image size (28x28)
    out_features: 10  # 10 digit classes

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001

  flows:
    train:
      _target_: lighter.flow.Flow
      batch: ["input", "target"]
      model: ["input"]
      criterion: ["pred", "target"]

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 64
      dataset:
        _target_: torchvision.datasets.MNIST
        root: ./data
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Lambda
              lambd: "$lambda x: x.view(-1)"  # Flatten 28x28 to 784
```

Now run it:

```bash
lighter fit quick_start.yaml
```

**Congratulations!** 🎉 You just trained a neural network using only YAML configuration. No Python classes, no training loops, just declarative config.

## Understanding What Just Happened

Let's break down the magic. Every Lighter config has **two essential parts**:

### 1. The `trainer` Section - *How* to Train

```yaml
trainer:
  _target_: pytorch_lightning.Trainer  # Use PyTorch Lightning's Trainer
  max_epochs: 3                        # Train for 3 epochs
```

This is pure [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html). Any Trainer argument works here:

- `max_epochs` - How many times to iterate through the dataset
- `devices` - Number of GPUs/TPUs to use
- `precision` - Enable mixed precision training
- `callbacks` - Add custom callbacks

**You already know this part** if you've used PyTorch Lightning!

### 2. The `system` Section - *What* to Train

```yaml
system:
  _target_: lighter.System  # Lighter's orchestrator

  model: ...       # Your neural network
  criterion: ...   # Loss function
  optimizer: ...   # Learning algorithm
  dataloaders: ... # Data loading
```

This is Lighter's magic. The `System` class orchestrates everything, so you don't write training steps manually.

## The Secret Sauce: `_target_`

Every component in your config uses `_target_` to specify which Python class to instantiate:

```yaml
model:
  _target_: torch.nn.Linear  # "Create a torch.nn.Linear layer"
  in_features: 784           # First argument
  out_features: 10           # Second argument
```

**This is equivalent to:**

```python
model = torch.nn.Linear(in_features=784, out_features=10)
```

**The pattern:**

- `_target_` → Which class to use
- Other keys → Constructor arguments for that class

You can use **any** Python class this way:

- PyTorch models: `torchvision.models.resnet18`
- Custom models: `my_project.models.CustomNet`
- Third-party libraries: `timm.create_model`

## Connecting Components with `@`

How does the optimizer get the model's parameters? The `@` reference syntax:

```yaml
optimizer:
  _target_: torch.optim.Adam
  params: "$@system#model.parameters()"  # ← This!
  lr: 0.001
```

**Breaking it down:**

- `system#model` - Fetch the model definition from the system section using `#`
- `@` - Instantiate that model
- `.parameters()` - Call its `.parameters()` method using `$` that lets you run Python code

**Common reference patterns:**

```yaml
# Reference the model parameters
"$@system#model.parameters()"

# Reference the optimizer (for schedulers)
"@system#optimizer"

# Reference training dataloader
"@system#dataloaders#train"
```

!!! tip "Learn More"
    For advanced configuration details and the differences between `#`, `@`, `$`, and `%`, see the [Configuration Guide](../how-to/configure.md).

## Your Superpower: CLI Overrides

Want to experiment with different hyperparameters? **Don't edit the YAML.** Override from the command line:

```bash
# Try different learning rates
lighter fit quick_start.yaml --system#optimizer#lr=0.01

# Train longer
lighter fit quick_start.yaml --trainer#max_epochs=10

# Bigger batch size
lighter fit quick_start.yaml --system#dataloaders#train#batch_size=128

# Combine multiple overrides
lighter fit quick_start.yaml \
  --trainer#max_epochs=10 \
  --system#optimizer#lr=0.001 \
  --trainer#devices=2
```

**This is how you run experiments.** Keep your base config clean, tweak parameters on the fly.

## Level Up: Real Model, Real Dataset

Let's upgrade to a proper computer vision experiment:

```yaml title="image_classifier.yaml"
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System

  model:
    _target_: torchvision.models.resnet18  # Real CNN!
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001

  flows:
    train:
      _target_: lighter.flow.Flow
      batch: ["input", "target"]
      model: ["input"]
      criterion: ["pred", "target"]
    val:
      _target_: lighter.flow.Flow
      batch: ["input", "target"]
      model: ["input"]
      criterion: ["pred", "target"]

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 32
      shuffle: true
      num_workers: 4  # Parallel data loading
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ./data
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomHorizontalFlip
            - _target_: torchvision.transforms.RandomCrop
              size: 32
              padding: 4
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.4914, 0.4822, 0.4465]
              std: [0.2470, 0.2435, 0.2616]

    val:
      _target_: torch.utils.data.DataLoader
      batch_size: 64
      shuffle: false
      num_workers: 4
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ./data
        train: false
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.4914, 0.4822, 0.4465]
              std: [0.2470, 0.2435, 0.2616]
```

Run it:

```bash
lighter fit image_classifier.yaml
```

**You're now training ResNet-18 on CIFAR-10** with data augmentation, validation splits, and parallel data loading. All from config!

## Quick Reference: The Syntax

You've learned the essential syntax. Here's your cheat sheet:

| Symbol | Meaning | Example |
|--------|---------|---------|
| `_target_` | Which class to instantiate | `_target_: torch.nn.Linear` |
| `@` | Reference to instantiated object | `"@system#model"` |
| `$` | Execute Python expression | `"$@system#model.parameters()"` |
| `#` | Navigate config hierarchy | `system#optimizer#lr` |
| `.` | Access Python attributes | `model.parameters()` |

**The pattern to remember:**

```yaml
component_name:
  _target_: path.to.ClassName
  constructor_arg1: value1
  constructor_arg2: value2
```

## You're Now a Lighter User! 🚀

In 15 minutes, you've learned to:

- ✅ Write declarative YAML configs instead of boilerplate Python
- ✅ Use `_target_` to instantiate any Python class
- ✅ Connect components with `@` references
- ✅ Override hyperparameters from the command line
- ✅ Train real models on real datasets

## What's Next?

You're ready for more advanced topics:

### Continue Learning

- **[Image Classification Tutorial](image_classification.md)** - Build a complete project with validation, callbacks, and logging
- **[Configuration Guide](../how-to/configure.md)** - Deep dive into advanced config patterns
- **[Flows](../how-to/flows.md)** - Lighter's secret weapon for task-agnostic training

### Quick Tips

- **Stuck?** Check the [Troubleshooting Guide](../how-to/troubleshooting.md)
- **Questions?** See the [FAQ](../faq.md)
- **Coming from PyTorch Lightning?** Read the [Migration Guide](../migration/from-pytorch-lightning.md)

### Try These Next

```bash
# Add TensorBoard logging
lighter fit config.yaml \
  --trainer#logger#_target_=pytorch_lightning.loggers.TensorBoardLogger \
  --trainer#logger#save_dir=logs

# Use multiple GPUs
lighter fit config.yaml --trainer#devices=2

# Enable mixed precision training
lighter fit config.yaml --trainer#precision="16-mixed"
```

**Welcome to config-driven deep learning!** 🎉

You're no longer writing boilerplate - you're declaring experiments. Every change is traceable, every run is reproducible, and every experiment is just a YAML file away.

Now go build something amazing! 💪
