import argparse
import os

from data import balanced_sample, split_train_val_test, save_jsonl
from config import PROFILES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="Path to ExtraHop JSONL(.gz)"
    )
    ap.add_argument("--per-class", type=int, default=250_000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--profile", default="tiny", choices=list(PROFILES.keys()))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prof = PROFILES[args.profile]

    X, y = balanced_sample(
        args.input, per_class=args.per_class, seed=args.seed
    )
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_train_val_test(X, y)

    os.makedirs(args.out, exist_ok=True)
    save_jsonl(os.path.join(args.out, "train.jsonl"), Xtr, ytr, prof.max_len)
    save_jsonl(os.path.join(args.out, "val.jsonl"), Xva, yva, prof.max_len)
    save_jsonl(os.path.join(args.out, "test.jsonl"), Xte, yte, prof.max_len)

    print(f"Saved to {args.out}/train.jsonl, val.jsonl, test.jsonl")
