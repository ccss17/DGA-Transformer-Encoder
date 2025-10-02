"""DGA Detection Model using Transformer Encoder.

This model treats domain names as sequences of characters and uses a Transformer
encoder to learn patterns that distinguish DGA (algorithmically generated) domains
from legitimate ones.

Key design decisions:
1. Character-level tokenization: Captures subword patterns that LSTMs miss
   - DGAs often have unusual character n-grams (e.g., "xkwj", "qmzo")
   - Character level avoids OOV issues with new DGA families

2. Pre-LN Transformer: Modern architecture that's easier to train
   - More stable gradients than Post-LN (original Transformer)
   - No need for learning rate warmup
   - Can go deeper without tricks

3. [CLS] token pooling: Standard approach for sequence classification
   - Transformer learns to aggregate sequence info into [CLS]
   - Better than mean/max pooling empirically

4. Learned positional embeddings: Domain structure is important
   - TLD patterns (last few chars)
   - Subdomain patterns (first few chars)
   - Learned embeddings capture this better than fixed sinusoids for short seqs
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from .charset import PAD, VOCAB_SIZE
from .config import PROFILES

NUM_CLASSES = 2


# ------------------------------
# Core encoder (Pre-LayerNorm)
# ------------------------------
class DGAEncoder(nn.Module):
    """
    Transformer encoder for DGA (Domain Generation Algorithm) detection.

    Architecture overview:
    1. Token + Position embeddings
    2. Transformer encoder (Pre-LN variant)
    3. Classification head on [CLS] token

    Design choices:
    - Pre-LN (Layer Norm before attention): More stable training, doesn't need warmup
    - Positional embeddings (learned): Capture character position importance
    - [CLS] token pooling: Standard for sequence classification, better than mean pooling
    """

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
    ) -> None:
        super().__init__()

        # Token embeddings: Convert character IDs to dense vectors
        # padding_idx=PAD tells the embedding to zero out padding tokens
        # This prevents the model from learning anything from pad tokens
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)

        # Positional embeddings: Learned position encodings (not sinusoidal)
        # Each position gets its own learned embedding vector
        # For domain names, position matters (e.g., TLD vs subdomain patterns)
        self.pos = nn.Embedding(max_len, d_model)

        # Register position IDs as a buffer (not a parameter, but moves with model to GPU)
        # This is just [0, 1, 2, ..., max_len-1] repeated for batching
        self.register_buffer(
            "position_ids",
            torch.arange(max_len).unsqueeze(0),
            persistent=False,  # Don't save in checkpoint, we can recreate it
        )

        # Transformer Encoder Layer with Pre-LN architecture
        # Pre-LN (norm_first=True) is more stable than Post-LN:
        # - Gradients flow better (less vanishing gradient issues)
        # - No need for learning rate warmup
        # - Can train deeper models without special initialization tricks
        #
        # ffn_mult=4 means FFN hidden dim = 4 * d_model (standard Transformer ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,  # Expect input as (batch, seq, features)
            norm_first=True,  # Pre-LN: LayerNorm before attention (more stable!)
        )

        # Stack multiple encoder layers
        # Each layer does: Self-Attention -> FFN
        # With Pre-LN, each sublayer is: LN -> Sublayer -> Residual
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Final LayerNorm on [CLS] token output
        # This normalizes the representation before classification
        # Helps with training stability and generalization
        self.norm = nn.LayerNorm(d_model)

        # Classification head: Simple linear layer
        # Maps [CLS] representation (d_model) to class logits (NUM_CLASSES)
        # No activation here - we'll use CrossEntropyLoss which applies softmax
        self.clf = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        x: (B, L) token ids with CLS at index 0

        Steps:
        1. Look up token embeddings and add positional embeddings
        2. Pass through transformer encoder layers
        3. Extract [CLS] token (position 0) and normalize
        4. Project to class logits
        """
        b, L = x.shape  # b = batch size, L = sequence length

        # Expand position IDs to match batch size
        # pos will be [[0,1,2,...,L-1], [0,1,2,...,L-1], ...] for batch
        pos = self.position_ids[:, :L].expand(b, L)

        # Token + position embeddings
        # This is element-wise addition (broadcasting works because both are (B, L, d_model))
        # Each position gets its own learned offset added to the token embedding
        h = self.tok(x) + self.pos(pos)  # h = hidden states (embeddings)

        # Pass through transformer encoder
        # Self-attention allows each character to attend to all other characters
        # This captures long-range dependencies (e.g., suffix patterns, character distributions)
        h = self.enc(h)  # h = transformed hidden states

        # Extract and normalize the [CLS] token representation
        # [CLS] is always at position 0 in our encoding scheme
        # The transformer has learned to aggregate sequence information into [CLS]
        cls = self.norm(
            h[:, 0]
        )  # cls = normalized [CLS] token (sequence representation)

        # Project to class logits (benign vs DGA)
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

    This enables:
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
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Forward pass compatible with HF Trainer.

        Args:
            input_ids: Token IDs (B, L) with CLS at index 0
            attention_mask: Not used (padding handled by PAD token automatically)
            labels: Ground truth labels for classification (B,)
            return_dict: Whether to return SequenceClassifierOutput

        Returns:
            SequenceClassifierOutput or tuple with loss and logits

        Note on loss computation:
        - CrossEntropyLoss combines LogSoftmax + NLLLoss
        - It expects raw logits (no softmax applied) and class indices
        - Automatically handles the softmax internally for numerical stability
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # Forward through the existing encoder
        # This calls DGAEncoder.forward() which returns (B, NUM_CLASSES) logits
        logits = self.encoder(input_ids)

        # Compute loss if labels provided (training mode)
        # CrossEntropyLoss expects:
        #   - Input: (N, C) where C is number of classes
        #   - Target: (N,) with class indices in [0, C-1]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )

        # Return format depends on return_dict flag
        # HF Trainer expects return_dict=True by default
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # Could add intermediate layer outputs here
            attentions=None,  # Could add attention weights here for visualization
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


__all__ = [
    "DGAEncoderConfig",
    "DGAEncoderForSequenceClassification",
    "build_model",
]
