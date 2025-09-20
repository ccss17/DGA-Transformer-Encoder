# atomic save/load, RNG & scaler/opt states
from __future__ import annotations
import os, tempfile, random
import torch, numpy as np

DEFAULT_LAST = "last.pt"


def atomic_save(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def capture_rng_state() -> dict:
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    try:
        random.setstate(state["py"])
        np.random.set_state(state["np"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


def save_checkpoint(path: str, *, model, optimizer, scaler, epoch: int, global_step: int,
                    best_f1: float, sampler_state: dict, config: dict, size: str = "tiny") -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_f1": float(best_f1),
        "sampler": sampler_state or {},
        "size": size,
        "config": config,
        "rng": capture_rng_state(),
    }
    atomic_save(ckpt, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")