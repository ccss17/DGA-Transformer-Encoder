import json
import gzip
import random
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
        Expected keys: {"domain": <str>, "threat": "benign"|"dga"}
        First few lines:
    # COPYRIGHT 2023 BY EXTRAHOP NETWORKS, INC.
    {"domain": "eohcbdibsjoiafxnrvddh", "threat": "dga"}
    {"domain": "myhandeczema", "threat": "benign"}
    """
    with gzip.open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            j = json.loads(line)
            domain = j["domain"]
            threat = j["threat"]
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
    
    rng = random.Random(seed)
    for label, domains in by_label.items():
        selected = rng.sample(domains, per_class)
        samples.extend(selected)
        labels.extend([label] * len(selected))

    samples, labels = shuffle(samples, labels, random_state=seed)
    return samples, labels


def save_jsonl(path: str, X: List[str], y: List[int], max_len: int):
    """Saves with HF Trainer-compatible field names (input_ids, labels).
    Example: save_jsonl("out.jsonl", ["abc"], [1], 4) writes {"domain": "abc", ...}.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for domain, label in zip(X, y):
            obj = {
                "domain": domain,
                "labels": int(label),
                "input_ids": encode_domain(domain, max_len),
            }
            f.write(json.dumps(obj) + "\n")


def create_dataset(
    input: str,
    per_class: int = 250_000,
    out: str = "data",
    seed: int = 0,
):
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
    typer.run(create_dataset)
