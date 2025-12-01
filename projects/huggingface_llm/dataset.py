from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer_name, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load the dataset
        self.dataset = load_dataset(dataset_name, split=split)

        # Reduce dataset size for faster demo
        if "train" in split:
            self.dataset = self.dataset.shuffle(seed=42).select(range(2000))
        else:
            self.dataset = self.dataset.shuffle(seed=42).select(range(500))

        # Tokenize the dataset
        self.dataset = self.dataset.map(
            lambda e: self.tokenizer(e["text"], truncation=True, padding="max_length", max_length=max_length), batched=True
        )

        # Rename 'label' to 'labels' for Hugging Face model compatibility
        self.dataset = self.dataset.rename_column("label", "labels")

        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
