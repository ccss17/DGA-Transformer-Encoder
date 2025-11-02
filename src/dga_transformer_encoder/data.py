"""Convert JSONL data to HuggingFace Dataset format."""

from pathlib import Path

from datasets import load_dataset, DatasetDict
import torch

__all__ = [
    "prepare_datasets",
    "create_data_collator",
]


def prepare_datasets(
    data_dir: str | Path = "data",
) -> DatasetDict:
    """Load train/val/test splits as HuggingFace DatasetDict.
    """
    data_dir = Path(data_dir)
    datasets = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
            "test": str(data_dir / "test.jsonl"),
        }
    )
    
    # Set format to PyTorch tensors for efficiency
    datasets.set_format("torch", columns=["input_ids", "labels"])
    return datasets


def create_data_collator():
    def collate_fn(examples):
        """domains are already padded to max_len in the JSONL files"""
        batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "labels": torch.tensor(
                [ex["labels"] for ex in examples], dtype=torch.long
            ),
        }
        return batch

    return collate_fn
