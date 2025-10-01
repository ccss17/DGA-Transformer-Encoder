"""DGA Transformer Encoder - Domain Generation Algorithm Classifier.
Uses a character-level Transformer encoder to distinguish DGA (algorithmically
generated) domains from legitimate ones.
"""

from .model import (
    DGAEncoder,
    DGAEncoderConfig,
    DGAEncoderForSequenceClassification,
    build_model,
)
from .data import (
    load_jsonl_as_dataset,
    prepare_datasets,
    create_data_collator,
)
from .charset import (
    VOCAB_SIZE,
    PAD,
    CLS,
    encode_domain,
    normalize_domain,
)
from .config import PROFILES

__all__ = [
    "DGAEncoder",
    "DGAEncoderConfig",
    "DGAEncoderForSequenceClassification",
    "build_model",
    "load_jsonl_as_dataset",
    "prepare_datasets",
    "create_data_collator",
    "VOCAB_SIZE",
    "PAD",
    "CLS",
    "encode_domain",
    "normalize_domain",
    "PROFILES",
]
