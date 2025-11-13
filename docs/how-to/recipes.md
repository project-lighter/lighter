---
title: Configuration Recipes
---

# Configuration Recipes

Ready-to-use configurations for common scenarios. Copy, adapt, and run.

## Training Infrastructure

### Multi-GPU: DDP

```yaml
trainer:
  devices: -1  # All GPUs (or specify: devices: 4)
  accelerator: gpu
  strategy: ddp
  precision: "16-mixed"
  max_epochs: 100
  sync_batchnorm: true  # Synchronize batch norm across GPUs

system:
  dataloaders:
    train:
      batch_size: 32  # Per GPU! Effective = 32 * num_gpus
      num_workers: 4  # Per GPU
      pin_memory: true
      persistent_workers: true
```

### Multi-GPU: DeepSpeed (Large Models)

```yaml
trainer:
  devices: -1
  accelerator: gpu
  strategy: deepspeed_stage_2  # or deepspeed_stage_3
  precision: "16-mixed"

system:
  dataloaders:
    train:
      batch_size: 8  # Smaller for large models
```

### Multi-GPU: FSDP (Very Large Models)

```yaml
trainer:
  devices: -1
  accelerator: gpu
  strategy: fsdp
  precision: "bf16-mixed"  # BFloat16 often better for FSDP
```

### Experiment Tracking: TensorBoard

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: logs
    name: my_experiment
    version: null  # Auto-increment

  callbacks:
    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: step
```

### Experiment Tracking: Weights & Biases

```yaml
_requires_:
  - "$import datetime"

trainer:
  logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    project: my_project
    name: "$'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')"
    save_dir: logs
    log_model: true  # Save checkpoints to W&B

  callbacks:
    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: epoch
```

### Multiple Loggers

```yaml
trainer:
  logger:
    - _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: logs
      name: experiment
    - _target_: pytorch_lightning.loggers.WandbLogger
      project: my_project
      name: experiment
    - _target_: pytorch_lightning.loggers.CSVLogger
      save_dir: logs
```

### Best Model Checkpointing

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: checkpoints
      filename: "best-{epoch:02d}-{val_loss:.4f}"
      monitor: val_loss
      mode: min
      save_top_k: 3  # Keep best 3
      save_last: true
```

### Monitor Multiple Metrics

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: checkpoints/loss
      filename: "loss-{epoch:02d}-{val_loss:.4f}"
      monitor: val_loss
      mode: min
      save_top_k: 2

    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: checkpoints/acc
      filename: "acc-{epoch:02d}-{val_acc:.4f}"
      monitor: val_acc
      mode: max
      save_top_k: 2
```

### Early Stopping

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val_loss
      patience: 10
      mode: min
      min_delta: 0.001
      verbose: true

    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val_loss
      mode: min
      save_top_k: 1
```

## Data Augmentation

### Image Classification

```yaml
system:
  dataloaders:
    train:
      dataset:
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224
              scale: [0.8, 1.0]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.RandomRotation
              degrees: 15
            - _target_: torchvision.transforms.ColorJitter
              brightness: 0.4
              contrast: 0.4
              saturation: 0.4
              hue: 0.1
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]

    val:
      dataset:
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256
            - _target_: torchvision.transforms.CenterCrop
              size: 224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
```

### RandAugment

```yaml
_requires_:
  - "$from torchvision.transforms import RandAugment"

system:
  dataloaders:
    train:
      dataset:
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224
            - _target_: torchvision.transforms.RandAugment
              num_ops: 2
              magnitude: 9
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
```

## Learning Rate Schedules

### Cosine Annealing

```yaml
vars:
  max_epochs: 100

system:
  optimizer:
    _target_: torch.optim.AdamW
    params: "$@system::model.parameters()"
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@system::optimizer"
    T_max: "%vars::max_epochs"
    eta_min: 0.00001
```

### ReduceLROnPlateau

```yaml
system:
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    optimizer: "@system::optimizer"
    mode: min
    factor: 0.5
    patience: 5
    min_lr: 0.00001
    verbose: true
```

### Step Decay

```yaml
system:
  scheduler:
    _target_: torch.optim.lr_scheduler.MultiStepLR
    optimizer: "@system::optimizer"
    milestones: [30, 60, 90]
    gamma: 0.1
```

## Transfer Learning

### Fine-tuning Pretrained Models

```yaml
system:
  model:
    _target_: torchvision.models.resnet50
    weights: IMAGENET1K_V2
    num_classes: 10

  optimizer:
    _target_: torch.optim.SGD
    params: "$@system::model.parameters()"
    lr: 0.001  # Lower LR for fine-tuning
    momentum: 0.9

  callbacks:
    - _target_: lighter.callbacks.Freezer
      modules: ["layer1", "layer2"]  # Freeze early layers
```

### Differential Learning Rates

```yaml
system:
  model:
    _target_: torchvision.models.resnet50
    weights: IMAGENET1K_V2
    num_classes: 10

  optimizer:
    _target_: torch.optim.SGD
    params:
      - params: "$@system::model.layer1.parameters()"
        lr: 0.0001
      - params: "$@system::model.layer4.parameters()"
        lr: 0.001
      - params: "$@system::model.fc.parameters()"
        lr: 0.01
    momentum: 0.9
```

## Gradient Handling

### Gradient Clipping

```yaml
trainer:
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm  # or 'value'
```

### Gradient Accumulation

```yaml
trainer:
  accumulate_grad_batches: 4

system:
  dataloaders:
    train:
      batch_size: 8  # Effective: 8 * 4 = 32
```

## Performance Optimization

### Fast Training

```yaml
trainer:
  precision: "16-mixed"
  devices: -1
  accelerator: gpu
  benchmark: true  # cudnn.benchmark

system:
  dataloaders:
    train:
      num_workers: 8
      pin_memory: true
      persistent_workers: true
      prefetch_factor: 2
      batch_size: 64
```

### Memory Optimization

```yaml
trainer:
  precision: "16-mixed"
  accumulate_grad_batches: 8

system:
  dataloaders:
    train:
      batch_size: 4  # Effective: 4 * 8 = 32
      num_workers: 2
```

## Development & Debugging

### Fast Development Run

```yaml
trainer:
  fast_dev_run: true  # 1 batch of train/val/test
```

### Overfit Single Batch

```yaml
trainer:
  overfit_batches: 1
  max_epochs: 100
  logger: false
```

### Profiling

```yaml
trainer:
  profiler: simple  # or 'advanced', 'pytorch'
  max_epochs: 1
  limit_train_batches: 10
```

---

## Machine Learning Paradigms

### Multi-Task Learning

Train one model on multiple tasks with shared representations.

```yaml
system:
  model:
    _target_: my_project.MultiTaskModel
    backbone: resnet50
    num_classes_classification: 10
    num_classes_segmentation: 2

  criterion:
    _target_: my_project.MultiTaskLoss
    classification_weight: 1.0
    segmentation_weight: 0.5

  metrics:
    train:
      classification:
        - _target_: torchmetrics.Accuracy
          task: multiclass
          num_classes: 10
      segmentation:
        - _target_: torchmetrics.JaccardIndex
          task: binary
    val: "%system::metrics::train"
```

```python title="my_project/losses.py"
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, classification_weight=1.0, segmentation_weight=1.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_weight = classification_weight
        self.seg_weight = segmentation_weight

    def forward(self, pred, target):
        cls_loss = self.cls_loss(pred['classification'], target['class'])
        seg_loss = self.seg_loss(pred['segmentation'], target['mask'])

        return {
            "total": self.cls_weight * cls_loss + self.seg_weight * seg_loss,
            "classification": cls_loss,
            "segmentation": seg_loss,
        }
```

All sublosses logged automatically.

### Self-Supervised Learning (Contrastive)

```yaml
system:
  model:
    _target_: my_project.SimCLRModel
    backbone: resnet18
    projection_dim: 128

  criterion:
    _target_: my_project.NTXentLoss
    temperature: 0.5

  adapters:
    train:
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0
        target_accessor: null  # No labels
```

```python title="my_project/model.py"
import torch.nn as nn

class SimCLRModel(nn.Module):
    def __init__(self, backbone='resnet18', projection_dim=128):
        super().__init__()
        self.encoder = torch.hub.load('pytorch/vision', backbone, weights=None)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projector(features)
```

### Knowledge Distillation

Train small student model from large teacher.

```yaml
system:
  model:
    _target_: my_project.DistillationModel
    student:
      _target_: torchvision.models.resnet18
      num_classes: 10
    teacher:
      _target_: torchvision.models.resnet50
      num_classes: 10
    teacher_weights: checkpoints/teacher.ckpt

  criterion:
    _target_: my_project.DistillationLoss
    temperature: 3.0
    alpha: 0.7  # Distillation loss weight
```

```python title="my_project/model.py"
import torch.nn as nn

class DistillationModel(nn.Module):
    def __init__(self, student, teacher, teacher_weights=None):
        super().__init__()
        self.student = student
        self.teacher = teacher

        if teacher_weights:
            self.teacher.load_state_dict(torch.load(teacher_weights))

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, x):
        student_logits = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        return {"student": student_logits, "teacher": teacher_logits}
```

### Curriculum Learning

Progressively increase task difficulty.

```python title="my_project/model.py"
class CurriculumModel(nn.Module):
    def __init__(self, num_classes=10, max_epochs=100):
        super().__init__()
        self.backbone = nn.Sequential(...)
        self.classifier = nn.Linear(512, num_classes)
        self.max_epochs = max_epochs

    def forward(self, x, epoch=None):
        # Lighter automatically injects epoch
        if epoch is not None:
            difficulty = min(epoch / self.max_epochs, 1.0)
            x = self.apply_difficulty(x, difficulty)

        features = self.backbone(x)
        return self.classifier(features)
```

No config needed—epoch injection is automatic.

### Model Ensembling

```yaml
system:
  model:
    _target_: my_project.EnsembleModel
    models:
      - _target_: torchvision.models.resnet18
        num_classes: 10
      - _target_: torchvision.models.resnet34
        num_classes: 10
    checkpoints:
      - checkpoints/model1.ckpt
      - checkpoints/model2.ckpt

  inferer:
    _target_: my_project.EnsembleInferer
    ensemble_method: "average"  # or "voting"
```

```python title="my_project/model.py"
class EnsembleInferer:
    def __init__(self, ensemble_method="average"):
        self.method = ensemble_method

    def __call__(self, x, model, **kwargs):
        predictions = []
        for m in model.models:
            m.eval()
            with torch.no_grad():
                predictions.append(m(x))

        if self.method == "average":
            return torch.stack(predictions).mean(dim=0)
        elif self.method == "voting":
            return torch.stack(predictions).mode(dim=0)[0]
```

### Cross-Validation

```python title="cross_validation.py"
import subprocess

def run_kfold(base_config="config.yaml", k=5):
    for fold in range(k):
        subprocess.run([
            "lighter", "fit", base_config,
            f"system::dataloaders::train::dataset::fold={fold}",
            f"trainer::logger::version=fold_{fold}"
        ])

if __name__ == "__main__":
    run_kfold()
```

```yaml title="config.yaml"
system:
  dataloaders:
    train:
      dataset:
        _target_: my_project.KFoldDataset
        k: 5
        fold: 0  # Overridden from CLI
```

---

## Production Setup

Complete production-ready configuration:

```yaml
_requires_:
  - "$import datetime"

vars:
  experiment_name: "production_run"
  timestamp: "$datetime.datetime.now().strftime('%Y%m%d_%H%M%S')"

trainer:
  _target_: pytorch_lightning.Trainer
  devices: -1
  accelerator: gpu
  strategy: ddp
  precision: "16-mixed"
  max_epochs: 200
  gradient_clip_val: 1.0

  logger:
    - _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: logs
      name: "%vars::experiment_name"
      version: "%vars::timestamp"
    - _target_: pytorch_lightning.loggers.WandbLogger
      project: production
      name: "$%vars::experiment_name + '_' + %vars::timestamp"

  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: "$'checkpoints/' + %vars::experiment_name"
      filename: "best-{epoch:02d}-{val_loss:.4f}"
      monitor: val_loss
      mode: min
      save_top_k: 3
      save_last: true

    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val_loss
      patience: 20
      mode: min

    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: epoch

system:
  _target_: lighter.System

  model:
    _target_: torchvision.models.resnet50
    weights: IMAGENET1K_V2
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.AdamW
    params: "$@system::model.parameters()"
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@system::optimizer"
    T_max: 200

  dataloaders:
    train:
      batch_size: 64
      num_workers: 8
      pin_memory: true
      persistent_workers: true
    val:
      batch_size: 128
      num_workers: 4
      pin_memory: true
```

## Next Steps

- [Configuration Reference](configuration.md) - Complete syntax guide
- [Troubleshooting](troubleshooting.md) - Debug issues
- [Adapters](adapters.md) - Handle any data format
- [System Internals](../design/system.md) - Understanding the pipeline
