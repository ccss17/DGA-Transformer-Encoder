# StatefulRandomSampler for step-level resume
from __future__ import annotations
import torch
from torch.utils.data import Sampler
from typing import Iterator

class StatefulRandomSampler(Sampler[int]):
    """Random sampler that can save/restore its iteration position.
    It creates a deterministic permutation per epoch from (seed + epoch),
    and tracks the current cursor (pos). Safe to resume mid-epoch.
    Note: If DataLoader drop_last=True, resuming mid-batch will move to the next batch boundary.
    """
    def __init__(self, data_source, *, seed: int = 42, epoch: int = 0,
                 start_pos: int = 0) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.pos = int(start_pos)
        self._build_perm()

    def _build_perm(self) -> None:
        n = len(self.data_source)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.perm = torch.randperm(n, generator=g).tolist()
        # clamp start position if needed
        if self.pos > len(self.perm):
            self.pos = len(self.perm)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.pos = 0
        self._build_perm()

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        self.seed = int(state.get("seed", self.seed))
        self.epoch = int(state.get("epoch", self.epoch))
        self.pos = int(state.get("pos", 0))
        self._build_perm()

    def __iter__(self) -> Iterator[int]:
        # Advance cursor lazily, DataLoader will fetch from here
        while self.pos < len(self.perm):
            idx = self.perm[self.pos]
            self.pos += 1
            yield idx

    def __len__(self) -> int:
        return len(self.data_source)