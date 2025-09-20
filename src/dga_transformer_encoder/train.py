# resume & intervals
import os
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from .config import PROFILES
from .charset import VOCAB_SIZE
from .model import build_dga_encoder
from .sampler import StatefulRandomSampler
from .checkpoint import save_checkpoint, load_checkpoint, DEFAULT_LAST


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.X, self.y = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.X.append(j["ids"])  # pre-encoded ids
                self.y.append(j["label"])  # 0=benign, 1=dga

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(
            self.y[i], dtype=torch.long
        )


@torch.no_grad()
def eval_macro_f1(
    model: nn.Module, loader: DataLoader, device: str = "cuda"
) -> float:
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = model(x).argmax(1).cpu().tolist()
        preds += p
        gts += y.tolist()
    return f1_score(gts, preds, average="macro")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument(
        "--size", default="tiny", choices=["tiny", "small", "medium", "heavy"]
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--save-dir", default="ckpt")
    ap.add_argument("--seed", type=int, default=42)
    # Resume & intervals
    ap.add_argument(
        "--resume",
        nargs="?",
        const=DEFAULT_LAST,
        default=None,
        help="path to checkpoint (default: save-dir/last.pt)",
    )
    ap.add_argument("--ckpt-interval-steps", type=int, default=200)
    ap.add_argument("--ckpt-interval-seconds", type=int, default=300)
    ap.add_argument(
        "--save-best",
        action="store_true",
        help="also save best.pt when val improves",
    )
    args = ap.parse_args()

    prof = PROFILES[args.size]
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets
    tr = JsonlDataset(os.path.join(args.data, "train.jsonl"))
    va = JsonlDataset(os.path.join(args.data, "val.jsonl"))
    dl_va = DataLoader(
        va,
        batch_size=max(1024, args.batch_size * 4),
        num_workers=2,
        pin_memory=True,
    )

    # Model/optim/loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_dga_encoder(
        args.size, vocab_size=VOCAB_SIZE, max_len=prof.max_len
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=prof.lr,
        betas=(0.9, 0.98),
        weight_decay=prof.weight_decay,
    )
    loss = nn.CrossEntropyLoss(label_smoothing=prof.label_smoothing)

    # FlashAttention-2 / SDPA
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Resume state
    start_epoch, global_step, best_f1 = 0, 0, -1.0
    sampler_state = {"seed": args.seed, "epoch": 0, "pos": 0}
    last_path = os.path.join(args.save_dir, DEFAULT_LAST)
    resume_path = args.resume
    if resume_path is not None and not os.path.isabs(resume_path):
        resume_path = os.path.join(args.save_dir, resume_path)
    if resume_path and os.path.exists(resume_path):
        ck = load_checkpoint(resume_path)
        model.load_state_dict(ck["state_dict"])
        if ck.get("optimizer") is not None:
            opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler") is not None:
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception:
                pass
        start_epoch = int(ck.get("epoch", 0))
        global_step = int(ck.get("global_step", 0))
        best_f1 = float(ck.get("best_f1", -1.0))
        sampler_state = ck.get("sampler", sampler_state)
        # RNG state
        from checkpoint import restore_rng_state

        restore_rng_state(ck.get("rng", {}))
        print(
            f"[resume] epoch={start_epoch}, global_step={global_step}, best_f1={best_f1:.4f}"
        )

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    t_last = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Sampler with state
        if sampler_state.get("epoch", 0) != epoch:
            sampler_state = {"seed": args.seed, "epoch": epoch, "pos": 0}
        sampler = StatefulRandomSampler(
            tr,
            seed=sampler_state["seed"],
            epoch=epoch,
            start_pos=sampler_state.get("pos", 0),
        )
        dl_tr = DataLoader(
            tr,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )

        model.train()
        for x, y in dl_tr:
            x, y = (
                x.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x)
                l = loss(logits, y)
            scaler.scale(l).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            global_step += 1

            # Periodic checkpoint (time- or step-based)
            if (global_step % max(1, args.ckpt - interval - steps) == 0) or (
                time.time() - t_last >= args.ckpt_interval_seconds
            ):
                sampler_state = sampler.state_dict()
                save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=opt,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_f1=best_f1,
                    sampler_state=sampler_state,
                    config={
                        "d_model": prof.d_model,
                        "nhead": prof.nhead,
                        "layers": prof.layers,
                        "ffn_mult": prof.ffn_mult,
                        "max_len": prof.max_len,
                        "dropout": prof.dropout,
                        "num_classes": 2,
                    },
                    size=args.size,
                )
                t_last = time.time()

        # End of epoch: validate and save
        f1 = eval_macro_f1(model, dl_va, device)
        print(f"epoch {epoch}: macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            if args.save_best:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "size": args.size,
                        "config": {
                            "d_model": prof.d_model,
                            "nhead": prof.nhead,
                            "layers": prof.layers,
                            "ffn_mult": prof.ffn_mult,
                            "max_len": prof.max_len,
                            "dropout": prof.dropout,
                            "num_classes": 2,
                        },
                    },
                    os.path.join(args.save_dir, f"{args.size}_best.pt"),
                )
        # Save last at epoch boundary (pos resets to 0 next epoch)
        sampler_state = {"seed": args.seed, "epoch": epoch + 1, "pos": 0}
        save_checkpoint(
            last_path,
            model=model,
            optimizer=opt,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            best_f1=best_f1,
            sampler_state=sampler_state,
            config={
                "d_model": prof.d_model,
                "nhead": prof.nhead,
                "layers": prof.layers,
                "ffn_mult": prof.ffn_mult,
                "max_len": prof.max_len,
                "dropout": prof.dropout,
                "num_classes": 2,
            },
            size=args.size,
        )
