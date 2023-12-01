# Using Lighter in your own projects

With Lighter, you can be as hands-on as you wish when using it in your project. For example, you can use a pre-defined Lighter configuration and,

- [x] Train on your own dataset
- [x] Train on your data + Add a custom model architecture
- [x] Train on your data + Add a custom model architecture + Add a complex loss function
- [x] Customization per your imagination! 

Lets start by looking at each of these one by one. At the end of this, you will hopefully have a better idea of how best you can leverage lighter

### Training on your own dataset

If you are reproducing another study you often start with a pre-defined configuration. Let us take the case of `cifar10.yaml` shown in [Quickstart](./quickstart.md). You have a dataset of Chest X-rays that you want to use to reproduce the same training that was done on CIFAR10. With lighter, all you need to change is the highlighted sections.

```yaml title="cifar10.yaml"  hl_lines="18-29"
system:
  _target_: lighter.LighterSystem
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

To replace this with your own dataset, you can create a Pytorch dataset that produces images and targets in the same format as torchvision datasets, i.e (image, target) tuple.

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
        sample = {'image': image, 'target': target}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample['image'], sample['target']

```

!!! note
    Lighter works with the default torchvision format of (image, target) and also with `dict` with `input` and `target` keys. The input/target key or tuple can contain complex input/target organization, e.g. multiple images for input and multiple labels for target


Now that you have built your dataset, all you need to do is add it to the lighter config! But wait, how will Lighter know where your code is? All lighter configs contain a `project` key that takes the full path to where your python code is located. Once you set this up, call `project.MyXRayDataset` and Lighter will pick up the dataset. 

In the above example, the path of the dataset is `/home/user/project/my_xray_dataset.py`. Copy the config shown above, make the following changes and run on the terminal

<div class="annotate" markdown>
=== "xray.yaml"


    ```yaml hl_lines="1" hl_lines="1 20-23"
    project: /home/user/project (1)
    system:
    _target_: lighter.LighterSystem
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
        _target_: project.MyXRayDataset
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
    lighter fit --config_file xray.yaml
    ```

</div>   

1. Make sure to put an `__init__.py` file in this directory. Remember this is needed for an importable python module