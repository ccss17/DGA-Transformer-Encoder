import os
import json
import gzip
import random
from typing import Iterable, Tuple, List

from charset import encode_domain

LABEL_MAP = {"benign": 0, "dga": 1}


def stream_extrahop(path: str) -> Iterable[Tuple[str, int]]:
    """Yield (domain, label) from ExtraHop JSONL.GZ file.
    Expected keys: {"domain": <str>, "threat": "benign"|"dga"}
    Domain should NOT contain TLD.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            j = json.loads(line)
            d = j.get("domain", "")
            t = j.get("threat", "")
            if t not in LABEL_MAP:  # skip unknown
                continue
            yield d, LABEL_MAP[t]


def dedup_pairs(pairs: Iterable[Tuple[str, int]]):
    seen = set()
    for d, y in pairs:
        key = (d, y)
        if key in seen:  # exact duplicate
            continue
        seen.add(key)
        yield d, y


def balanced_sample(
    path: str, per_class: int, seed: int = 0
) -> Tuple[List[str], List[int]]:
    random.seed(seed)
    bins = {0: [], 1: []}
    for d, y in stream_extrahop(path):
        if len(bins[y]) < per_class:
            bins[y].append(d)
        if all(len(b) >= per_class for b in bins.values()):
            break
    X = bins[0] + bins[1]
    y = [0] * len(bins[0]) + [1] * len(bins[1])
    # shuffle
    idx = list(range(len(X)))
    random.shuffle(idx)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    return X, y


def split_train_val_test(
    X: List[str], y: List[int], ratios=(0.8, 0.1, 0.1), seed=42
):
    random.seed(seed)
    n = len(X)
    idx = list(range(n))
    random.shuffle(idx)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    tr_idx, va_idx, te_idx = idx[:a], idx[a:b], idx[b:]

    def take(idxs):
        return [X[i] for i in idxs], [y[i] for i in idxs]

    return take(tr_idx), take(va_idx), take(te_idx)


def save_jsonl(path: str, X: List[str], y: List[int], max_len: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d, t in zip(X, y):
            obj = {
                "domain": d,
                "label": int(t),
                "ids": encode_domain(d, max_len),
            }
            f.write(json.dumps(obj) + "\n")
