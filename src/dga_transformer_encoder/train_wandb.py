# W&B-integrated variant with resume support
import os, json, argparse, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from config import PROFILES
from charset import VOCAB_SIZE
from model import build_dga_encoder
from sampler import StatefulRandomSampler
from checkpoint import save_checkpoint, load_checkpoint, DEFAULT_LAST

class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.X, self.y = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.X.append(j["ids"])  # pre-encoded ids
                self.y.append(j["label"])  # 0/1
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)

@torch.no_grad()
def eval_macro_f1(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = model(x).argmax(1).cpu().tolist()
        preds += p; gts += y.tolist()
    return f1_score(gts, preds, average="macro")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--size", default="tiny", choices=["tiny","small","medium"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--save-dir", default="ckpt")
    ap.add_argument("--seed", type=int, default=42)
    # Resume & intervals
    ap.add_argument("--resume", nargs="?", const=DEFAULT_LAST, default=None)
    ap.add_argument("--ckpt-interval-steps", type=int, default=200)
    ap.add_argument("--ckpt-interval-seconds", type=int, default=300)
    ap.add_argument("--save-best", action="store_true")
    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="dga-transformer")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-offline", action="store_true")
    ap.add_argument("--wandb-id", default=None, help="set to resume a prior run id")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--wandb-artifact", action="store_true")
    args = ap.parse_args()

    prof = PROFILES[args.size]
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets & loaders
    tr = JsonlDataset(os.path.join(args.data, "train.jsonl"))
    va = JsonlDataset(os.path.join(args.data, "val.jsonl"))
    dl_va = DataLoader(va, batch_size=max(1024, args.batch_size*4), num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_dga_encoder(args.size, vocab_size=VOCAB_SIZE, max_len=prof.max_len).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=prof.lr, betas=(0.9, 0.98), weight_decay=prof.weight_decay)
    loss = nn.CrossEntropyLoss(label_smoothing=prof.label_smoothing)

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ---- Resume checkpoint ----
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
        from checkpoint import restore_rng_state
        restore_rng_state(ck.get("rng", {}))
        print(f"[resume] epoch={start_epoch}, global_step={global_step}, best_f1={best_f1:.4f}")

    # ---- W&B init (after possible resume) ----
    use_wandb = args.wandb
    if use_wandb:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        import wandb
        if args.wandb_id:
            os.environ["WANDB_RESUME"] = "allow"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.tags,
            id=args.wandb_id,
            config={
                "size": args.size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": prof.lr,
                "weight_decay": prof.weight_decay,
                "label_smoothing": prof.label_smoothing,
                "d_model": prof.d_model,
                "nhead": prof.nhead,
                "layers": prof.layers,
                "ffn_mult": prof.ffn_mult,
                "max_len": prof.max_len,
                "dataset_train": len(tr),
                "dataset_val": len(va),
                "resume_epoch": start_epoch,
                "resume_global_step": global_step,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)
        wandb.summary["params"] = int(sum(p.numel() for p in model.parameters()))
    else:
        run = None

    best_path = os.path.join(args.save_dir, f"{args.size}_best.pt")
    t_last = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Sampler per epoch with position resume
        if sampler_state.get("epoch", 0) != epoch:
            sampler_state = {"seed": args.seed, "epoch": epoch, "pos": 0}
        sampler = StatefulRandomSampler(tr, seed=sampler_state["seed"], epoch=epoch, start_pos=sampler_state.get("pos", 0))
        dl_tr = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)

        model.train()
        step_loss_sum, step = 0.0, 0
        t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x)
                l = loss(logits, y)
            scaler.scale(l).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            global_step += 1
            step += 1
            step_loss_sum += float(l.detach())

            # periodic save
            if (global_step % max(1, args.ckpt_interval_steps) == 0) or (time.time() - t_last >= args.ckpt_interval_seconds):
                sampler_state = sampler.state_dict()
                save_checkpoint(
                    last_path,
                    model=model, optimizer=opt, scaler=scaler,
                    epoch=epoch, global_step=global_step,
                    best_f1=best_f1, sampler_state=sampler_state,
                    config={
                        "d_model": prof.d_model, "nhead": prof.nhead, "layers": prof.layers,
                        "ffn_mult": prof.ffn_mult, "max_len": prof.max_len, "dropout": prof.dropout,
                        "num_classes": 2,
                    }, size=args.size,
                )
                t_last = time.time()
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": float(l.detach()), "global_step": global_step})

        # validate
        epoch_time = time.time() - t0
        f1 = eval_macro_f1(model, dl_va, device)
        avg_loss = step_loss_sum / max(1, step)
        print(f"epoch {epoch}: macro-F1={f1:.4f} | train_loss={avg_loss:.4f} | time_sec={epoch_time:.1f}")
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "val/macro_f1": f1,
                "train/loss_epoch": avg_loss,
                "time/epoch_sec": epoch_time,
                "lr": opt.param_groups[0]["lr"],
                "global_step": global_step,
            })

        # best & last
        if f1 > best_f1:
            best_f1 = f1
            if args.save_best:
                torch.save({
                    "state_dict": model.state_dict(),
                    "size": args.size,
                    "config": {
                        "d_model": prof.d_model, "nhead": prof.nhead, "layers": prof.layers,
                        "ffn_mult": prof.ffn_mult, "max_len": prof.max_len, "dropout": prof.dropout,
                        "num_classes": 2,
                    }
                }, best_path)
                if use_wandb and args.wandb_artifact:
                    import wandb
                    art = wandb.Artifact(name=f"{args.size}-best", type="model")
                    art.add_file(best_path)
                    wandb.log_artifact(art)

        # epoch-boundary last save (pos resets)
        sampler_state = {"seed": args.seed, "epoch": epoch+1, "pos": 0}
        save_checkpoint(
            last_path,
            model=model, optimizer=opt, scaler=scaler,
            epoch=epoch+1, global_step=global_step,
            best_f1=best_f1, sampler_state=sampler_state,
            config={
                "d_model": prof.d_model, "nhead": prof.nhead, "layers": prof.layers,
                "ffn_mult": prof.ffn_mult, "max_len": prof.max_len, "dropout": prof.dropout,
                "num_classes": 2,
            }, size=args.size,
        )

    if use_wandb:
        import wandb
        wandb.summary["best_val_macro_f1"] = float(best_f1)
        wandb.finish()