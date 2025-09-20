"""
DGA Transformer Encoders (Tiny/Small/Medium)

Drop this file in place of your existing model.py.
Compatible with the previously shared train.py/eval.py (which pass
layer/nhead/d_model via args/config). You now also get convenience
classes for explicit sizes and a simple factory.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass

# ------------------------------
# Core encoder (Pre-LayerNorm)
# ------------------------------
class DGAEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_len: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.clf = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) token ids with CLS at index 0."""
        b, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(b, L)
        h = self.tok(x) + self.pos(pos)
        h = self.enc(h)
        cls = self.norm(h[:, 0])
        return self.clf(cls)

# ------------------------------------
# Size presets (Tiny / Small / Medium)
# ------------------------------------
class TinyDGAEncoder(DGAEncoder):
    def __init__(self, *, vocab_size: int, max_len: int = 64, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=256,
            nhead=8,
            nlayers=4,
            num_classes=num_classes,
            dropout=dropout,
            ffn_mult=4,
        )

class SmallDGAEncoder(DGAEncoder):
    def __init__(self, *, vocab_size: int, max_len: int = 64, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=384,
            nhead=6,
            nlayers=6,
            num_classes=num_classes,
            dropout=dropout,
            ffn_mult=4,
        )

class MediumDGAEncoder(DGAEncoder):
    def __init__(self, *, vocab_size: int, max_len: int = 64, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=512,
            nhead=8,
            nlayers=8,
            num_classes=num_classes,
            dropout=dropout,
            ffn_mult=4,
        )

# ------------------------------
# Simple factory & config helper
# ------------------------------
@dataclass
class EncoderConfig:
    d_model: int = 256
    nhead: int = 8
    layers: int = 4
    ffn_mult: int = 4
    max_len: int = 64
    dropout: float = 0.1
    num_classes: int = 2


def build_dga_encoder(size: str, *, vocab_size: int, max_len: int = 64, num_classes: int = 2, dropout: float = 0.1) -> DGAEncoder:
    size = size.lower()
    if size == "tiny":
        return TinyDGAEncoder(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes, dropout=dropout)
    if size == "small":
        return SmallDGAEncoder(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes, dropout=dropout)
    if size == "medium":
        return MediumDGAEncoder(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"unknown size: {size}")


def encoder_from_config(cfg: dict, *, vocab_size: int) -> DGAEncoder:
    """Create encoder from a dict like the one saved in ckpt['config'].
    Expected keys: d_model, nhead, layers, ffn_mult, max_len, (optional) dropout, num_classes.
    """
    ec = EncoderConfig(**{k: cfg.get(k, getattr(EncoderConfig, k)) for k in EncoderConfig.__annotations__.keys()})
    return DGAEncoder(
        vocab_size=vocab_size,
        max_len=ec.max_len,
        d_model=ec.d_model,
        nhead=ec.nhead,
        nlayers=ec.layers,
        dropout=ec.dropout,
        ffn_mult=ec.ffn_mult,
        num_classes=ec.num_classes,
    )

__all__ = [
    "DGAEncoder",
    "TinyDGAEncoder",
    "SmallDGAEncoder",
    "MediumDGAEncoder",
    "EncoderConfig",
    "build_dga_encoder",
    "encoder_from_config",
]
