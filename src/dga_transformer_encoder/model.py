from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from .charset import PAD, VOCAB_SIZE
from .config import PROFILES

__all__ = [
    "DGAEncoderConfig",
    "DGAEncoderForSequenceClassification",
    "build_model",
]

NUM_CLASSES = 2


class DGAEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_len: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()

        # Token embeddings: Convert character IDs to dense vectors
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)

        # Positional embeddings: Learned position encodings 
        self.pos = nn.Embedding(max_len, d_model)

        # Register position IDs as a buffer (not a parameter, but moves with model to GPU)
        # This is just [0, 1, 2, ..., max_len-1] repeated for batching
        self.register_buffer(
            "position_ids",
            torch.arange(max_len).unsqueeze(0),
            persistent=False,  # Don't save in checkpoint, just recreate it
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,  # Expect input (batch, seq, features)
            norm_first=True,  # Pre-LN: LayerNorm before attention 
        )

        # Stack multiple encoder layers
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Final LayerNorm on [CLS] token output
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.clf = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) token ids with CLS at index 0
        """
        b, L = x.shape  # b = batch size, L = sequence length

        # Expand position IDs to match batch size
        # pos will be [[0,1,2,...,L-1], [0,1,2,...,L-1], ...] for batch
        pos = self.position_ids[:, :L].expand(b, -1)

        # Token + position embeddings
        h = self.tok(x) + self.pos(pos)

        # Pass through transformer encoder
        h = self.enc(h)

        # Extract and normalize the [CLS] token representation
        cls = self.norm(h[:, 0])  

        # Project to class logits (benign vs DGA)
        # No activation here - we'll use CrossEntropyLoss which applies softmax
        return self.clf(cls)


class DGAEncoderConfig(PretrainedConfig):
    """Configuration for DGAEncoder compatible with HuggingFace Transformers.

    can be saved/loaded using HF's standard save_pretrained()
    and from_pretrained() methods.
    """

    model_type = "dga_encoder"

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.ffn_mult = ffn_mult
        self.num_labels = NUM_CLASSES  # Use global constant


class DGAEncoderForSequenceClassification(PreTrainedModel):
    """HuggingFace-compatible wrapper around DGAEncoder.

    - Automatic checkpoint management via Trainer
    - save_pretrained() / from_pretrained() methods
    - Integration with HF ecosystem (datasets, evaluate, etc.)
    - W&B logging via Trainer's report_to="wandb"
    """

    config_class = DGAEncoderConfig

    def __init__(self, config: DGAEncoderConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DGAEncoder(
            vocab_size=config.vocab_size,
            max_len=config.max_len,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            ffn_mult=config.ffn_mult,
        )

        # Initialize weights (HF convention)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass compatible with HF Trainer.

        input_ids: Token IDs (B, L) with CLS at index 0
        attention_mask: Not used (padding handled by PAD token automatically)
        labels: Ground truth labels for classification (B,)

        Returns:
            SequenceClassifierOutput with loss and logits
        """
        # DGAEncoder.forward() returns (B, NUM_CLASSES) logits
        logits = self.encoder(input_ids)

        # Compute loss if labels provided (training mode)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def build_model(size: str = "tiny") -> DGAEncoderForSequenceClassification:
    """
    model = build_model("tiny")
    model.save_pretrained("./my_model")
    loaded = DGAEncoderForSequenceClassification.from_pretrained("./my_model")
    """
    prof = PROFILES[size]
    config = DGAEncoderConfig(
        vocab_size=VOCAB_SIZE,
        max_len=prof.max_len,
        d_model=prof.d_model,
        nhead=prof.nhead,
        num_layers=prof.num_layers,
        dropout=prof.dropout,
        ffn_mult=prof.ffn_mult,
    )
    return DGAEncoderForSequenceClassification(config)
