"""Vision-Language datasets for CLIP-style training.

Real datasets for vision-language learning:
- Flickr30k: 31k images with 5 captions each
- COCO Captions: 330k images with captions
- Flickr8k: 8k images with 5 captions each (smaller, easier to obtain)
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch.utils.data import Dataset


class Caption(TypedDict):
    image_path: Path
    caption: str


class Flickr30kDataset(Dataset):
    """Flickr30k dataset for vision-language training.

    Flickr30k contains 31,000 images with 5 captions each.
    This loader reads the supplied corpus. Provide a separately partitioned root
    for each population; it does not invent a training/validation split.

    Download:
    1. Request access: https://shannon.cs.illinois.edu/DenotationGraph/
    2. Extract images to root/flickr30k-images/
    3. Download captions file to root/results_20130124.token

    Args:
        root: Root directory containing images and captions.
        tokenizer: HuggingFace tokenizer for text encoding.
        max_length: Maximum text sequence length.
        image_transform: Transform for images.
    """

    def __init__(
        self,
        root: str,
        tokenizer: Callable[..., dict[str, torch.Tensor]] | None = None,
        max_length: int = 77,
        image_transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        if not callable(tokenizer):
            raise ValueError("A callable tokenizer is required; configure a tokenizer matching the text encoder vocabulary")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform

        # Load captions
        self.data = self._load_captions()

        if len(self.data) == 0:
            raise FileNotFoundError(
                f"No data found for Flickr30k dataset at '{self.root}'.\n\n"
                f"Please download the dataset:\n"
                f"1. Request access: https://shannon.cs.illinois.edu/DenotationGraph/\n"
                f"2. Extract images to: {self.root}/flickr30k-images/\n"
                f"3. Download captions to: {self.root}/results_20130124.token\n\n"
                f"Or use Flickr8kDataset for a smaller, easier-to-obtain alternative."
            )

    def _load_captions(self) -> list[Caption]:
        """Load image-caption pairs."""
        caption_file = self.root / "results_20130124.token"
        image_dir = self.root / "flickr30k-images"

        data: list[Caption] = []
        if caption_file.exists():
            with open(caption_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        image_id = parts[0].split("#")[0]
                        caption = parts[1]
                        image_path = image_dir / image_id
                        if image_path.exists():
                            data.append({"image_path": image_path, "caption": caption})
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # Load image
        image = self._load_image(item["image_path"])
        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize text
        encoding = self.tokenizer(
            item["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": item["caption"],
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from file."""
        from torchvision.io import read_image

        return cast(torch.Tensor, read_image(str(path))).float() / 255.0


class Flickr8kDataset(Dataset):
    """Flickr8k dataset for vision-language training.

    Flickr8k contains 8,000 images with 5 captions each.
    Smaller and easier to obtain than Flickr30k.

    Download from Kaggle:
    https://www.kaggle.com/datasets/adityajn105/flickr8k

    Args:
        root: Root directory containing images and captions.
        split: Dataset split ('train', 'val', 'test'). A corresponding split
            manifest is required; available manifests must be disjoint.
        tokenizer: HuggingFace tokenizer for text encoding.
        max_length: Maximum text sequence length.
        image_transform: Transform for images.
    """

    # Standard Flickr8k splits
    SPLITS = {
        "train": "Flickr_8k.trainImages.txt",
        "val": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        tokenizer: Callable[..., dict[str, torch.Tensor]] | None = None,
        max_length: int = 77,
        image_transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        if not callable(tokenizer):
            raise ValueError("A callable tokenizer is required; configure a tokenizer matching the text encoder vocabulary")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform

        self.data = self._load_data()

        if len(self.data) == 0:
            raise FileNotFoundError(
                f"No data found for Flickr8k dataset at '{self.root}'.\n\n"
                "Please download the dataset from:\n"
                "https://www.kaggle.com/datasets/adityajn105/flickr8k\n\n"
                "Expected structure:\n"
                f"  {self.root}/Images/           (image files)\n"
                f"  {self.root}/captions.txt      (caption file)"
            )

    def _load_split_images(self) -> set[str]:
        """Require an explicit, nonempty, disjoint image-level split manifest."""
        if self.split not in self.SPLITS:
            raise ValueError(f"Unknown Flickr8k split {self.split!r}; choose train, val or test")
        split_file = self.root / self.SPLITS[self.split]
        if not split_file.is_file():
            raise FileNotFoundError(f"Required split manifest is missing: {split_file}; provide image names, one per line")
        images = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not images:
            raise ValueError(f"Split manifest is empty: {split_file}")
        for other, filename in self.SPLITS.items():
            path = self.root / filename
            if other != self.split and path.is_file():
                other_images = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
                overlap = images & other_images
                if overlap:
                    raise ValueError(f"Flickr8k {self.split} and {other} split manifests overlap: {sorted(overlap)[:5]}")
        return images

    def _load_data(self) -> list[Caption]:
        """Read real captions and selected images; missing inputs are errors."""
        caption_file = self.root / "captions.txt"
        image_dir = self.root / "Images"
        if not caption_file.exists():
            caption_file = self.root / "Flickr8k.token.txt"
        if not image_dir.exists():
            image_dir = self.root / "Flickr8k_Dataset"
        valid_images = self._load_split_images()
        data: list[Caption] = []
        with caption_file.open(encoding="utf-8", newline="") as stream:
            first_line = stream.readline()
            stream.seek(0)
            rows = (line.rstrip("\n").split("\t", 1) for line in stream) if "\t" in first_line else csv.reader(stream)
            for row in rows:
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError(f"Expected image and caption in {caption_file}, got {row!r}")
                if row[0] in {"image", "image_name"} and row[1] == "caption":
                    continue
                image_name, caption = row[0].split("#", 1)[0], row[1]
                if image_name not in valid_images:
                    continue
                image_path = image_dir / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(f"Selected Flickr8k image is missing: {image_path}")
                data.append({"image_path": image_path, "caption": caption})
        missing = valid_images - {item["image_path"].name for item in data}
        if missing:
            raise ValueError(f"Selected Flickr8k images have no captions: {sorted(missing)[:5]}")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        image = self._load_image(item["image_path"])
        if self.image_transform:
            image = self.image_transform(image)

        encoding = self.tokenizer(
            item["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": item["caption"],
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from file."""
        from torchvision.io import read_image

        return cast(torch.Tensor, read_image(str(path))).float() / 255.0


class COCOCaptionsDataset(Dataset):
    """COCO Captions dataset for vision-language training.

    COCO Captions: 330k images with 5 captions each.

    Download:
    1. Images: https://cocodataset.org/#download
    2. Annotations: Download 2017 Train/Val annotations

    Args:
        root: Root directory containing images.
        ann_file: Path to annotations JSON file.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum text sequence length.
        image_transform: Transform for images.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        tokenizer: Callable[..., dict[str, torch.Tensor]] | None = None,
        max_length: int = 77,
        image_transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.ann_file = Path(ann_file)
        if not callable(tokenizer):
            raise ValueError("A callable tokenizer is required; configure a tokenizer matching the text encoder vocabulary")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform

        self.data = self._load_annotations()

        if len(self.data) == 0:
            raise FileNotFoundError(
                f"No data found for COCO Captions dataset.\n\n"
                f"Images root: {self.root}\n"
                f"Annotations: {self.ann_file}\n\n"
                "Please download the dataset from:\n"
                "https://cocodataset.org/#download\n\n"
                "Expected files:\n"
                "  - train2017/ or val2017/ (image directories)\n"
                "  - annotations/captions_train2017.json"
            )

    def _load_annotations(self) -> list[Caption]:
        """Load COCO annotations."""
        import json

        data: list[Caption] = []
        if self.ann_file.exists():
            with open(self.ann_file) as f:
                coco = json.load(f)

            # Build image id to filename mapping
            id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

            # Load caption annotations
            for ann in coco["annotations"]:
                image_id = ann["image_id"]
                if image_id in id_to_file:
                    data.append(
                        {
                            "image_path": self.root / id_to_file[image_id],
                            "caption": ann["caption"],
                        }
                    )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        image = self._load_image(item["image_path"])
        if self.image_transform:
            image = self.image_transform(image)

        encoding = self.tokenizer(
            item["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from file."""
        from torchvision.io import read_image

        return cast(torch.Tensor, read_image(str(path))).float() / 255.0


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for vision-language batches."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }
