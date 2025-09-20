from dataclasses import dataclass


@dataclass
class Profile:
    layers: int
    d_model: int
    nhead: int
    ffn_mult: int
    dropout: float
    lr: float
    weight_decay: float
    label_smoothing: float
    max_len: int


PROFILES = {
    # ~3.2M params
    "tiny": Profile(
        layers=4,
        d_model=256,
        nhead=8,
        ffn_mult=4,
        dropout=0.1,
        lr=3e-4,
        weight_decay=0.01,
        label_smoothing=0.05,
        max_len=64,
    ),
    # ~10.8M params
    "small": Profile(
        layers=6,
        d_model=384,
        nhead=6,
        ffn_mult=4,
        dropout=0.1,
        lr=3e-4,
        weight_decay=0.01,
        label_smoothing=0.05,
        max_len=64,
    ),
    # ~25.6M params
    "medium": Profile(
        layers=8,
        d_model=512,
        nhead=8,
        ffn_mult=4,
        dropout=0.1,
        lr=2.5e-4,
        weight_decay=0.01,
        label_smoothing=0.05,
        max_len=64,
    ),
    # ~85.2M params (for strong GPUs)
    "heavy": Profile(
        layers=12,
        d_model=768,
        nhead=12,
        ffn_mult=4,
        dropout=0.1,
        lr=2.0e-4,
        weight_decay=0.01,
        label_smoothing=0.05,
        max_len=64,
    ),
}
