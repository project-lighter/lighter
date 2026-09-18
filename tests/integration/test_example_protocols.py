"""Tiny real fixtures verify example population and text semantics without downloads."""

import csv
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from torchvision.io import write_png

PROJECTS = Path(__file__).resolve().parents[2] / "projects"


def load_example(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", PROJECTS / name / "dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenize(text, *, max_length, **kwargs):
    ids = torch.tensor([ord(char) for char in text[:max_length]], dtype=torch.long)
    padded = torch.zeros(max_length, dtype=torch.long)
    padded[: len(ids)] = ids
    return {"input_ids": padded.unsqueeze(0), "attention_mask": (padded > 0).long().unsqueeze(0)}


@pytest.fixture
def flickr(tmp_path):
    (tmp_path / "Images").mkdir()
    module = load_example("vision_language")
    captions = {"train.png": 'A caption, with "quotes"', "val.png": "Different words", "test.png": "Held out image"}
    for index, name in enumerate(captions):
        write_png(torch.full((3, 4, 4), index + 10, dtype=torch.uint8), str(tmp_path / "Images" / name))
    with (tmp_path / "captions.txt").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "caption"])
        writer.writerows(captions.items())
    for split, filename in module.Flickr8kDataset.SPLITS.items():
        (tmp_path / filename).write_text(f"{split}.png\n")
    return module, tmp_path


def test_vision_language_requires_real_tokenizer_and_manifest(flickr):
    module, root = flickr
    with pytest.raises(ValueError, match="tokenizer"):
        module.Flickr8kDataset(str(root))
    (root / module.Flickr8kDataset.SPLITS["val"]).unlink()
    with pytest.raises(FileNotFoundError, match="split manifest"):
        module.Flickr8kDataset(str(root), split="val", tokenizer=tokenize)
    with pytest.raises(ValueError, match="Unknown.*split"):
        module.Flickr8kDataset(str(root), split="typo", tokenizer=tokenize)


def test_vision_language_populations_and_caption_tokens_are_distinct(flickr):
    module, root = flickr
    data = [module.Flickr8kDataset(str(root), split=split, tokenizer=tokenize) for split in ("train", "val", "test")]
    populations = [{item["image_path"].name for item in dataset.data} for dataset in data]
    assert populations == [{"train.png"}, {"val.png"}, {"test.png"}]
    assert data[0][0]["caption"] == 'A caption, with "quotes"'
    assert not torch.equal(data[0][0]["input_ids"], data[1][0]["input_ids"])
    assert data[0][0]["input_ids"].count_nonzero() > 0
    assert data[0][0]["image"].shape == (3, 4, 4)


def test_split_overlap_and_missing_selected_images_are_errors(flickr):
    module, root = flickr
    validation_manifest = root / module.Flickr8kDataset.SPLITS["val"]
    validation_manifest.write_text("train.png\n")
    with pytest.raises(ValueError, match="overlap"):
        module.Flickr8kDataset(str(root), tokenizer=tokenize)
    validation_manifest.write_text("val.png\n")
    (root / "Images" / "train.png").unlink()
    with pytest.raises(FileNotFoundError, match="Selected.*image"):
        module.Flickr8kDataset(str(root), tokenizer=tokenize)


def test_flickr30k_does_not_replace_corrupt_image_with_zeros(tmp_path):
    module = load_example("vision_language")
    (tmp_path / "flickr30k-images").mkdir()
    (tmp_path / "flickr30k-images" / "broken.jpg").write_text("not an image")
    (tmp_path / "results_20130124.token").write_text("broken.jpg#0\treal caption\n")
    dataset = module.Flickr30kDataset(str(tmp_path), tokenizer=tokenize)
    with pytest.raises(RuntimeError):
        dataset[0]


class TinyCIFAR(Dataset):
    def __init__(self, *, train, **kwargs):
        self.identities = list(range(20)) if train else list(range(100, 105))

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, index):
        identity = self.identities[index]
        return identity, identity % 2


@pytest.mark.parametrize("project", ["cifar10", "lora"])
def test_cifar_examples_reserve_official_test_population(monkeypatch, project):
    module = load_example(project)
    if project == "cifar10":
        monkeypatch.setattr(module, "CIFAR10", TinyCIFAR)
        factory = module.cifar10_split
    else:
        monkeypatch.setattr(module.datasets, "CIFAR100", TinyCIFAR)
        factory = module.CIFAR100Dataset
    before = torch.random.get_rng_state().clone()
    populations = []
    for split in ("train", "val", "test"):
        dataset = factory(root="unused", split=split)
        populations.append({dataset[index][0] for index in range(len(dataset))})
    assert len(populations[0]) == 18
    assert len(populations[1]) == 2
    assert populations[0].isdisjoint(populations[1])
    assert populations[0] | populations[1] == set(range(20))
    assert populations[2] == set(range(100, 105))
    assert populations[2].isdisjoint(populations[0] | populations[1])
    assert torch.equal(torch.random.get_rng_state(), before)
    again = factory(root="unused", split="val")
    assert {again[index][0] for index in range(len(again))} == populations[1]
