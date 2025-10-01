""" pixi run python src/dga_transformer_encoder/prepare_data.py \
        data/raw/dga-training-data-encoded.json.gz
"""

import json
import gzip
from pathlib import Path
from typing import Iterable, Tuple, List

import typer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .charset import encode_domain
from .config import PROFILES

LABEL_MAP = {"benign": 0, "dga": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
MAX_LEN = PROFILES["tiny"].max_len


def stream_dataset(path: str) -> Iterable[Tuple[str, int]]:
    """Yield (domain, label) pairs from an ExtraHop JSONL(.gz) file.

        Example: next(stream_dataset("data/raw/sample.jsonl.gz")) -> ("example", 1)
        Concept: supervised classification data ingestion for DGA detection.

        Expected keys: {"domain": <str>, "threat": "benign"|"dga"}
        Domain should NOT contain TLD.

        First few lines:
    # COPYRIGHT 2023 BY EXTRAHOP NETWORKS, INC.
    #
    # This file is subject to the terms and conditions defined in
    # file 'LICENSE', which is part of this source code package.
    #
    {"domain": "ocymmekqogkw", "threat": "dga"}
    {"domain": "eohcbdibsjoiafxnrvddh", "threat": "dga"}
    {"domain": "myhandeczema", "threat": "benign"}
    {"domain": "deummagdbawse", "threat": "dga"}
    {"domain": "vipozacyqexib", "threat": "dga"}
    {"domain": "gwirelessltd", "threat": "benign"}
    {"domain": "bankinvestmentaccount", "threat": "benign"}
    {"domain": "wmsbdckjyoq", "threat": "dga"}
    {"domain": "gmiuawqcygyyecackyo", "threat": "dga"}
    {"domain": "genositaliangrill", "threat": "benign"}
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            j = json.loads(line)
            domain = j.get("domain", "")
            threat = j.get("threat", "")
            if threat not in LABEL_MAP:  # skip unknown
                continue
            yield domain, LABEL_MAP[threat]


def balanced_sample(
    path: str, per_class: int, seed: int = 0
) -> Tuple[List[str], List[int]]:
    """Collect a balanced subset of domains for each label.

    Example: balanced_sample(path, per_class=1, seed=0) -> (["benign0", "dga0"], [0, 1])
    Concept: class balancing / stratified sampling prior to dataset splitting.
    """
    by_label = {label: [] for label in LABEL_MAP.values()}
    for domain, label in stream_dataset(path):
        by_label[label].append(domain)

    missing = [
        INV_LABEL_MAP[label]
        for label, domains in by_label.items()
        if len(domains) < per_class
    ]
    if missing:
        msg = ", ".join(missing)
        raise ValueError(
            f"Not enough samples to draw {per_class} items for labels: {msg}"
        )

    samples: List[str] = []
    labels: List[int] = []
    for label, domains in by_label.items():
        selected, _ = train_test_split(
            domains,
            train_size=per_class,
            random_state=seed + label,
            shuffle=True,
        )
        samples.extend(selected)
        labels.extend([label] * len(selected))

    samples, labels = shuffle(samples, labels, random_state=seed)
    return samples, labels


def save_jsonl(path: str, X: List[str], y: List[int], max_len: int):
    """Persist encoded domains and labels into a JSONL file.

    Example: save_jsonl("out.jsonl", ["abc"], [1], 4) writes {"domain": "abc", ...}.
    Concept: dataset serialization for downstream neural network training.

    Saves with HF Trainer-compatible field names (input_ids, labels).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for domain, label in zip(X, y):
            obj = {
                "domain": domain,
                "labels": int(label),  # plural for HF Trainer
                "input_ids": encode_domain(domain, max_len),
            }
            f.write(json.dumps(obj) + "\n")


def main(
    input: str,
    per_class: int = 250_000,
    out: str = "data",
    seed: int = 0,
) -> None:
    """Generate balanced train/val/test splits from ExtraHop data.

    Example: main("data/raw/dga-training-data-encoded.json.gz", per_class=10)
    Concept: CLI entry for dataset preparation prior to model training.
    """
    X, y = balanced_sample(input, per_class=per_class, seed=seed)
    Xtr, X_tmp, ytr, y_tmp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    Xva, Xte, yva, yte = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp,
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dir / "train.jsonl", Xtr, ytr, MAX_LEN)
    save_jsonl(out_dir / "val.jsonl", Xva, yva, MAX_LEN)
    save_jsonl(out_dir / "test.jsonl", Xte, yte, MAX_LEN)

    print(f"Saved to {out}/train.jsonl, val.jsonl, test.jsonl")


if __name__ == "__main__":
    typer.run(main)
