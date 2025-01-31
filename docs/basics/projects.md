# Using Lighter in your own projects

Lighter offers a flexible framework for integrating deep learning workflows into your projects. Whether you're starting with a pre-defined configuration or building a custom setup, Lighter adapts to your needs. Here’s how you can leverage Lighter:

- [x] Train on your own dataset
- [x] Train on your data + Add a custom model architecture
- [x] Train on your data + Add a custom model architecture + Add a complex loss function
- [x] Customization per your imagination! 

Let's start by looking at each of these one by one. At the end of this, you will hopefully have a better idea of how best you can leverage Lighter.

### Training on your own dataset

When reproducing a study or adapting a model to new data, you often start with a pre-defined configuration. For instance, consider the `cifar10.yaml` example from our [Quickstart](./quickstart.md). Suppose you have a dataset of Chest X-rays and wish to replicate the training process used for CIFAR10. With Lighter, you only need to modify specific sections of the configuration.

```yaml title="cifar10.yaml"  hl_lines="18-29"
system:
  _target_: lighter.System
  batch_size: 512

  model:
    _target_: torchvision.models.resnet18
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001

  datasets:
    train:
      _target_: torchvision.datasets.CIFAR10
      download: True
      root: .datasets
      train: True
      transform:
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
```

To integrate your dataset, create a PyTorch dataset class that outputs a dictionary with `input`, `target`, and optionally `id` keys. This ensures compatibility with Lighter's configuration system.

```py title="/home/user/project/my_xray_dataset.py"
class MyXRayDataset(Dataset):
    """
    Args:
    - root_dir (string): Directory with all the images.
    - annotations_file (string): Path to the annotations file.
    - transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir, annotations_file, transform=None):
        """
        Initialize the dataset.
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        target = self.annotations.iloc[idx, 1]
        sample = {'input': image, 'target': target}

        if self.transform:
            sample['input'] = self.transform(sample['input'])

        return sample

```

> **Note:** Lighter requires the dataset to return a dictionary with `input`, `target`, and optionally `id` keys. This format allows for complex input/target structures, such as multiple images or labels.


Once your dataset is ready, integrate it into the Lighter configuration. The `project` key in the config specifies the path to your Python code, allowing Lighter to locate and utilize your dataset. Simply reference your dataset class, and Lighter will handle the rest.

In the above example, the path of the dataset is `/home/user/project/my_xray_dataset.py`. Copy the config shown above, make the following changes and run on the terminal

<div class="annotate" markdown>
=== "xray.yaml"


    ```yaml hl_lines="1" hl_lines="1 20-23"
    project: /home/user/project (1)
    system:
    _target_: lighter.System
    batch_size: 512

    model:
        _target_: torchvision.models.resnet18
        num_classes: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001

    datasets:
        train:
        _target_: project.my_xray_dataset.MyXRayDataset
        root_dir: .
        annotations_file: label.csv
        transform:
            _target_: torchvision.transforms.Compose
            transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
                mean: [0.5, 0.5, 0.5]
                std: [0.5, 0.5, 0.5]
    ```
 

=== "Terminal"
    ```
    lighter fit --config xray.yaml
    ```

</div>   

1. Make sure to put an `__init__.py` file in this directory. Remember this is needed for an importable python module
