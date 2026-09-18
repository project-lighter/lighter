"""Prepare verified CIFAR bytes once; load only each requested population."""

import argparse
import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__:
    from .artifacts import sha256, write_json
else:
    from artifacts import sha256, write_json

UPSTREAM_MD5 = "c58f30108f718f92721af3b95e74349a"


def _read_batch(archive, name):
    member = archive.getmember(f"cifar-10-batches-py/{name}")
    if not member.isfile():
        raise ValueError(f"Expected a regular archive member: {name}")
    # The caller verifies the official archive digest before reading pickle.
    with archive.extractfile(member) as stream:
        value = pickle.load(stream, encoding="bytes")
    images = value[b"data"].reshape(-1, 3, 32, 32)
    labels = np.asarray(value[b"labels"], dtype=np.int64)
    if images.dtype != np.uint8 or images.shape != (len(labels), 3, 32, 32):
        raise ValueError("Unexpected CIFAR byte layout")
    if not np.all((labels >= 0) & (labels < 10)):
        raise ValueError("Unexpected CIFAR labels")
    return images, labels


def select_indices(labels, first, second=0, seed=1701):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected, following = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        if len(indices) < first + second:
            raise ValueError("Insufficient examples for the declared stratified selection")
        order = torch.randperm(len(indices), generator=generator).numpy()
        selected.extend(indices[order[:first]].tolist())
        following.extend(indices[order[first : first + second]].tolist())
    return sorted(selected), sorted(following)


def prepare(cache_root, output_dir):
    archive_path = Path(cache_root) / "cifar-10-python.tar.gz"
    digest = hashlib.md5(usedforsecurity=False)
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != UPSTREAM_MD5:
        raise ValueError("Archive is not the verified official CIFAR-10 Python archive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    # First collect labels, then read one 10,000-image batch at a time. Never
    # create separate full 50,000-image dataset objects for train and validation.
    train_labels = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for number in range(1, 6):
            images, labels = _read_batch(archive, f"data_batch_{number}")
            train_labels.extend(labels.tolist())
            del images
        test_images, test_labels = _read_batch(archive, "test_batch")
        test_indices, _ = select_indices(test_labels, 100, seed=1702)
        chosen_test = np.ascontiguousarray(test_images[test_indices])
        del test_images
    train_labels = np.asarray(train_labels)
    train_indices, val_indices = select_indices(train_labels, 200, 50, seed=1701)
    indices = {"train": train_indices, "val": val_indices, "test": test_indices}
    arrays = {
        "train": np.empty((2000, 3, 32, 32), dtype=np.uint8),
        "val": np.empty((500, 3, 32, 32), dtype=np.uint8),
        "test": chosen_test,
    }
    with tarfile.open(archive_path, "r:gz") as archive:
        for number in range(1, 6):
            images, _ = _read_batch(archive, f"data_batch_{number}")
            offset = (number - 1) * 10000
            for population in ("train", "val"):
                for destination, original in enumerate(indices[population]):
                    if offset <= original < offset + len(images):
                        arrays[population][destination] = images[original - offset]
            del images
    populations = {}
    for population in ("train", "val", "test"):
        source = "test" if population == "test" else "train"
        labels = test_labels if source == "test" else train_labels
        rows = [
            {"id": f"{source}:{index:05d}", "original_index": index, "label": int(labels[index]), "source_split": source}
            for index in indices[population]
        ]
        manifest_path = output_dir / f"{population}.json"
        image_path = output_dir / f"{population}.npy"
        write_json(manifest_path, {"schema_version": 1, "population": population, "source_split": source, "rows": rows})
        np.save(image_path, arrays[population], allow_pickle=False)
        populations[population] = {
            "manifest": manifest_path.name,
            "manifest_sha256": sha256(manifest_path),
            "images": image_path.name,
            "images_sha256": sha256(image_path),
            "count": len(rows),
        }
    pixel_hashes = {key: [hashlib.sha256(image.tobytes()).hexdigest() for image in values] for key, values in arrays.items()}
    duplicate_counts = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        counts = {}
        for digest in pixel_hashes[right]:
            counts[digest] = counts.get(digest, 0) + 1
        duplicate_counts[f"{left}:{right}"] = sum(counts.get(digest, 0) for digest in pixel_hashes[left])
    result = {
        "schema_version": 1,
        "dataset": "CIFAR10",
        "archive": {
            "path": str(archive_path.resolve()),
            "sha256": sha256(archive_path),
            "bytes": archive_path.stat().st_size,
            "upstream_md5": UPSTREAM_MD5,
        },
        "generation": {
            "torch_version": torch.__version__,
            "train_seed": 1701,
            "test_seed": 1702,
            "algorithm": "torch.randperm/private_cpu_generator/class_order_then_sort_original_indices",
        },
        "preprocessing": {"source": "uint8 RGB CHW", "operation": "float32 / 255", "augmentation": False},
        "duplicate_policy": {"action": "retain original indices", "cross_population_equal_pixel_pairs": duplicate_counts},
        "populations": populations,
    }
    write_json(output_dir / "data.json", result)
    return result


class ImagePopulation(Dataset):
    def __init__(self, data_manifest, population):
        data_manifest = Path(data_manifest)
        data = json.loads(data_manifest.read_text())
        entry = data["populations"][population]
        manifest_path = data_manifest.parent / entry["manifest"]
        image_path = data_manifest.parent / entry["images"]
        for path, expected in ((manifest_path, entry["manifest_sha256"]), (image_path, entry["images_sha256"])):
            if sha256(path) != expected:
                raise ValueError(f"Artifact digest mismatch: {path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["population"] != population:
            raise ValueError("Population manifest mismatch")
        self.rows = manifest["rows"]
        ids = [row["id"] for row in self.rows]
        if any(type(value) is not str for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Sample IDs must be unique exact strings")
        if any(type(row["label"]) is not int or not 0 <= row["label"] < 10 for row in self.rows):
            raise ValueError("Labels must be integers in [0, 9]")
        self.images = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if len(self.rows) != entry["count"] or self.images.shape != (len(self.rows), 3, 32, 32):
            raise ValueError("Images and manifest rows must align in RGB CHW layout")
        if self.images.dtype != np.uint8 or not self.images.flags.c_contiguous:
            raise ValueError("Stored images must be contiguous uint8")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = torch.from_numpy(np.array(self.images[index], copy=True)).to(torch.float32).div_(255)
        return {"image": image, "label": row["label"], "id": row["id"]}


class EpochOrder(Sampler):
    """Public native set_epoch selects a schedule independent of model RNG."""

    def __init__(self, rows, seed, batch_size, order_manifest=None, data_manifest=None):
        self.ids = [row["id"] for row in rows]
        self.seed = seed
        self.epoch = 0
        self.orders = None
        if order_manifest is not None:
            order = json.loads(Path(order_manifest).read_text())
            if order["data_manifest_sha256"] != sha256(data_manifest):
                raise ValueError("Order manifest references a different data manifest")
            if order["seed"] != seed or order["batch_size"] != batch_size:
                raise ValueError("Order manifest seed or batch size mismatch")
            lookup = {value: index for index, value in enumerate(self.ids)}
            self.orders = {}
            for epoch in order["epochs"]:
                batches = epoch["batches"]
                flat = [value for batch in batches for value in batch]
                if (
                    len(flat) != len(self.ids)
                    or len(set(flat)) != len(flat)
                    or set(flat) != set(self.ids)
                    or any(len(batch) != batch_size for batch in batches[:-1])
                    or not batches
                    or not 1 <= len(batches[-1]) <= batch_size
                ):
                    raise ValueError("Each epoch must contain every exact ID once, with a retained tail")
                if epoch["epoch"] in self.orders:
                    raise ValueError("Duplicate epoch in order manifest")
                self.orders[epoch["epoch"]] = [lookup[value] for value in flat]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.orders is not None:
            if self.epoch + 1 not in self.orders:
                raise ValueError(f"No frozen order for epoch {self.epoch + 1}")
            return iter(self.orders[self.epoch + 1])
        generator = torch.Generator(device="cpu").manual_seed(self.seed * 100000 + self.epoch + 1)
        return iter(torch.randperm(len(self.ids), generator=generator).tolist())

    def __len__(self):
        return len(self.ids)


class ComparisonData(pl.LightningDataModule):
    def __init__(self, data_manifest="data/data.json", order_manifest=None, seed=17, batch_size=64):
        super().__init__()
        self.data_manifest = data_manifest
        self.order_manifest = order_manifest
        self.seed = seed
        self.batch_size = batch_size
        self.populations = {}

    def setup(self, stage):
        requested = {"fit": ("train", "val"), "validate": ("val",), "test": ("test",), "predict": ("test",)}[stage]
        for population in requested:
            self.populations[population] = ImagePopulation(self.data_manifest, population)

    def train_dataloader(self):
        dataset = self.populations["train"]
        sampler = EpochOrder(dataset.rows, self.seed, self.batch_size, self.order_manifest, self.data_manifest)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.populations["val"], batch_size=self.batch_size, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.populations["test"], batch_size=self.batch_size, num_workers=0, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--cache-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(prepare(options.cache_root, options.output_dir), indent=2))
