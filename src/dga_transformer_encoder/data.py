"""Convert JSONL data to HuggingFace Dataset format."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import torch


def load_jsonl_as_dataset(path: str | Path) -> Dataset:
    """Load JSONL file as HuggingFace Dataset.
    Expects JSONL with fields: input_ids, labels, domain

    Returns:
        HF Dataset with columns: input_ids, labels, domain
    """
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    return Dataset.from_list(data)


def prepare_datasets(
    data_dir: str | Path = "data",
) -> DatasetDict:
    """Load train/val/test splits as HuggingFace DatasetDict.

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl

    Returns:
        DatasetDict with keys: train, validation, test

    datasets = prepare_datasets("data")
    print(datasets)
    DatasetDict({
        train: Dataset({
            features: ['input_ids', 'labels', 'domain'],
            num_rows: 100000
        })
        validation: Dataset({
            features: ['input_ids', 'labels', 'domain'],
            num_rows: 10000
        })
        test: Dataset({
            features: ['input_ids', 'labels', 'domain'],
            num_rows: 10000
        })
    })
    """
    data_dir = Path(data_dir)
    datasets = DatasetDict(
        {
            "train": load_jsonl_as_dataset(data_dir / "train.jsonl"),
            "validation": load_jsonl_as_dataset(data_dir / "val.jsonl"),
            "test": load_jsonl_as_dataset(data_dir / "test.jsonl"),
        }
    )

    # Set format to PyTorch tensors for efficiency
    datasets.set_format("torch", columns=["input_ids", "labels"])

    return datasets


def create_data_collator():
    """Create a simple data collator for fixed-length sequences.

    domains are already padded to max_len in the JSONL files,
    we just need a simple collator that stacks tensors.

    Returns:
        Callable that batches examples
    """

    def collate_fn(examples):
        """Stack pre-padded sequences into batches."""
        batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "labels": torch.tensor(
                [ex["labels"] for ex in examples], dtype=torch.long
            ),
        }
        return batch

    return collate_fn


__all__ = [
    "load_jsonl_as_dataset",
    "prepare_datasets",
    "create_data_collator",
]
