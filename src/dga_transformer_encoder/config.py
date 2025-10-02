from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Profile:
    # Transformer depth; shallow stacks limit overfitting on modest datasets
    # and keep inference latency low on edge hardware.
    num_layers: int
    # Hidden width sized to match domain-string complexity without paying for
    # unnecessary parameters.
    d_model: int
    # Multi-head attention count balances representational power against
    # compute footprint for GPU inference.
    nhead: int
    # Feed-forward expansion keeps the standard 4x ratio for strong accuracy
    # while maintaining throughput.
    ffn_mult: int = 4
    # Dropout of 10% reduces co-adaptation; higher rates slowed convergence in
    # experiments.
    dropout: float = 0.1
    # AdamW learning rate chosen from sweep; 3e-4 converged stably without
    # schedule tuning.
    lr: float = 3e-4
    # Weight decay curbs weight drift and improves generalization when
    # training on long-running streams.
    weight_decay: float = 0.01
    # Sequence length covers observed second-level domains with spare room for
    # rare longer names.
    max_len: int = 64


PROFILES = {
    # ~3.2M params - for quick testing
    "tiny": Profile(num_layers=4, d_model=256, nhead=8),
    # ~10.8M params - balanced performance
    "small": Profile(num_layers=6, d_model=384, nhead=6),
}
