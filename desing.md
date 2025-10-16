

## Architecture

```
Input Domain String
    ↓
Character Tokenization (custom charset)
    ↓
Embedding Layer (learnable character embeddings)
    ↓
Positional Encoding (learned embeddings, max_len=64)
    ↓
Transformer Encoder Stack (N layers)
    ↓
[CLS] Token Pooling (extract first token representation)
    ↓
Classification Head (Linear → 2 classes)
    ↓
Output: [P(legit), P(DGA)]
```

### Model Components

#### 1. Character-Level Tokenization

Unlike word-based models, we tokenize at the **character level** because:
- DGA domains don't form meaningful "words"
- Character patterns (vowel/consonant ratios, repeating patterns) are strong signals
- No vocabulary limitations

**Character set**: `[a-z0-9-_]` (40 tokens total)
- Letters: a-z (26)
- Digits: 0-9 (10)
- Special: `-`, `_` (2)
- Special tokens: `<PAD>`, `<CLS>` (2)

```python
# Example encoding (dots are removed during normalization)
"googlecom" → [1, 8, 16, 16, 8, 13, 6, 4, 16, 14, 0, ...]  # [CLS] + character IDs + [PAD]
```

#### 2. Embedding Layer

Converts each character token into a learned dense vector:

```python
vocab_size = 40  # character set size (38 chars + 2 special tokens)
d_model = 256    # embedding dimension (tiny model)

embedding = nn.Embedding(vocab_size, d_model)
```

`d_model` is the **dimensionality of the model's hidden representations** - the size of the vector that represents each character/token.

- **example** - when you process the domain `"googlecom"`:

```python
# Input: "googlecom" (9 characters)
# After tokenization: [1, 8, 16, 16, 8, 13, 6, 4, 16, 14]
#                      ↑  ↑   ↑   ↑  ↑   ↑  ↑  ↑   ↑   ↑
#                    [CLS] g  o   o  g   l  e  c   o   m

# After embedding layer (d_model=256):
# Each character becomes a 256-dimensional vector
[
  [0.12, -0.43, 0.89, ..., 0.56],  # [CLS] token → 256 numbers
  [0.34, 0.21, -0.67, ..., -0.12], # 'g' → 256 numbers
  [-0.45, 0.78, 0.23, ..., 0.91],  # 'o' → 256 numbers
  [-0.45, 0.78, 0.23, ..., 0.91],  # 'o' → 256 numbers (same char, same embedding)
  [0.34, 0.21, -0.67, ..., -0.12], # 'g' → 256 numbers
  [0.67, -0.23, 0.45, ..., 0.78],  # 'l' → 256 numbers
  [-0.12, 0.56, -0.34, ..., 0.23], # 'e' → 256 numbers
  [0.89, -0.67, 0.12, ..., -0.45], # 'c' → 256 numbers
  [-0.45, 0.78, 0.23, ..., 0.91],  # 'o' → 256 numbers
  [0.23, -0.12, 0.67, ..., -0.89], # 'm' → 256 numbers
]
# Shape: [batch_size, seq_len, d_model] = [batch_size, 10, 256]
```

Each character is now represented as a **256-dimensional vector** instead of just a single integer ID. These dense vectors are learned during training and capture semantic relationships between characters.

**learnable embeddings:**
- Similar characters (e.g., vowels) cluster together in embedding space
- The model learns that certain character combinations are more "suspicious"
- Dense vectors carry much more information than one-hot encodings

#### 3. Positional Encoding

Adds position information to embeddings using **learned positional embeddings**:

```python
# Learned position embeddings (not sinusoidal)
self.pos = nn.Embedding(max_len, d_model)
```

**Why learned positional embeddings instead of sinusoidal?**

1. **Short sequences**: Domain names are short (max 64 chars), unlike NLP sequences (512-4096 tokens). Learned embeddings can memorize position-specific patterns without overfitting.

2. **Position-dependent character patterns**: In domain names, character position matters:
   - **Beginning positions**: Legitimate domains often start with common patterns (e.g., `google`, `amazon`, `facebook`)
   - **End positions**: Last few characters may indicate word endings or common suffixes
   - **Middle positions**: DGA domains have unusual character distributions throughout
   - Learned embeddings can specialize for each position, while sinusoidal encodings treat all positions uniformly

3. **Character-level dependencies**: Unlike word tokens, characters have strong positional patterns:
   - Vowels are more common in certain positions in legitimate domains
   - DGA domains often have unusual vowel/consonant distributions by position
   - Character transitions (bigrams/trigrams) vary by position
   - Learned embeddings capture these task-specific patterns better

4. **Standard practice**: Modern transformers (BERT, GPT-2) use learned positional embeddings. The BERT paper reported that learned embeddings produced "slightly better results" compared to sinusoidal encodings for classification tasks. For short sequences (≤64 chars), learned embeddings are a reasonable default choice.

#### 4. Transformer Encoder

The core of the model. Each layer contains:

**a) Multi-Head Self-Attention**

Computes attention weights between all character positions:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **Q, K, V**: Queries, Keys, Values (linear projections of input)
- **Multi-head**: Run attention in parallel with different learned projections
- **Self-attention**: Each character attends to all other characters

**Why this matters for DGAs:**
- Detects unusual character co-occurrences (e.g., `xq`, `fgh`)
- Learns character distribution patterns (DGAs have unnatural letter combinations)
- Captures positional dependencies (e.g., certain characters appear more at domain beginnings vs endings)

**b) Feed-Forward Network**

After attention, each position is processed independently:

```python
FFN(x) = GELU(Linear1(x)) @ Linear2
```

- **Expansion ratio**: 4x (e.g., 256 → 1024 → 256)
- **GELU activation(`nn.TransformerEncoderLayer` default)**: Smooth, differentiable non-linearity
- Learns complex non-linear transformations of attention outputs

**c) Layer Normalization + Residual Connections**

We use **Pre-Layer Normalization** (Pre-LN) architecture:

```python
# Pre-LN: LayerNorm applied BEFORE sub-layer
x = x + Sublayer(LayerNorm(x))

# vs Post-LN (original Transformer): LayerNorm AFTER sub-layer
x = LayerNorm(x + Sublayer(x))
```

**Why Layer Normalization instead of Batch Normalization?**

1. **Sequence length independence**: LayerNorm normalizes across features for each sample independently, while BatchNorm normalizes across the batch dimension
   - BatchNorm breaks with variable sequence lengths (common in NLP)
   - LayerNorm works consistently regardless of batch size

2. **Better for transformers**: LayerNorm normalizes each token's feature vector independently
   - Each position in the sequence gets normalized on its own
   - No dependency on other samples in the batch

3. **Inference stability**: LayerNorm doesn't need running statistics (unlike BatchNorm)
   - No train/eval mode differences
   - More stable for single-sample inference

**Why Pre-LN instead of Post-LN?**

1. **Training stability**: Pre-LN provides more stable gradients
   - Gradients flow directly through residual connections without passing through normalization layers
   - Reduces gradient vanishing/exploding issues in deep networks

2. **No warmup needed**: Pre-LN can train with constant learning rate from the start
   - Post-LN typically requires learning rate warmup for stability
   - Simpler training schedule

3. **Better for deeper models**: Pre-LN scales better as you add more layers
   - Can train deeper transformers without special initialization tricks
   - More forgiving to hyperparameter choices

4. **Empirical performance**: Modern transformers (GPT-2, GPT-3, etc.) use Pre-LN
   - Consistently better results than Post-LN in practice
   - Easier to optimize

**Architecture components:**
- **LayerNorm**: Stabilizes training by normalizing activations across features
- **Residual**: Allows gradients to flow through deep networks via skip connections

#### 5. Pooling Strategy

After the encoder stack, we have a sequence of hidden states. We extract the **[CLS] token** representation:

```python
# Shape: [batch, seq_len, d_model] → [batch, d_model]
cls = self.norm(hidden_states[:, 0])  # Extract [CLS] token at position 0
```

**What is Pooling?**

**Pooling** is the process of aggregating a sequence of vectors into a single fixed-size vector for classification. After the transformer encoder, we have hidden states for every position: `[batch, seq_len, d_model]`. For binary classification (DGA vs legitimate), we need a single vector per domain: `[batch, d_model]`.

**What is [CLS] Token Pooling?**

[CLS] token pooling means extracting the hidden state of a special classification token (`<CLS>`) that's prepended to every input sequence:

1. **Input preparation**: Add `<CLS>` token at the beginning
   ```python
   # Domain: "googlecom" 
   # Token IDs: [1, 8, 16, 16, 8, 13, 6, 4, 16, 14, 0, ...]
   #             ↑
   #           [CLS] token (ID=1)
   ```

2. **Transformer processing**: Through self-attention, `<CLS>` attends to all other tokens
   ```python
   # [CLS] "looks at" all characters and learns to aggregate:
   # - Character distribution patterns
   # - Unusual n-gram combinations  
   # - Positional dependencies
   ```

3. **Extract representation**: After all layers, take position 0
   ```python
   cls_representation = hidden_states[:, 0]  # [batch, d_model]
   ```

4. **Classification**: Pass through linear layer
   ```python
   logits = classifier(cls_representation)  # [batch, 2]
   ```

**Why [CLS] Token Pooling?**

We chose [CLS] token pooling over alternatives (mean pooling, max pooling, last token):

1. **Learned aggregation**: The transformer learns to encode the entire sequence into `<CLS>` through attention
   - `<CLS>` can attend to all character positions simultaneously
   - The model optimizes `<CLS>` specifically for classification during training

2. **Standard practice**: Used in BERT and modern transformer classifiers
   - Proven effectiveness for sequence classification tasks
   - Well-studied and reliable approach

3. **Global context**: Through multi-head attention, `<CLS>` captures:
   - Short-range dependencies (adjacent characters)
   - Long-range dependencies (first and last characters)
   - Overall character distribution patterns

4. **Task-specific optimization**: Unlike mean pooling (treats all positions equally), `<CLS>` learns to emphasize important features for DGA detection

**Alternative Pooling Methods:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **[CLS] Token** (ours) | Extract position 0 | Learned aggregation, captures global context | Adds one extra token |
| **Mean Pooling** | Average all tokens | Simple, uses all positions | Requires padding mask, treats all positions equally |
| **Max Pooling** | Take maximum values | Captures strongest features | Loses positional info, sensitive to outliers |
| **Last Token** | Extract last non-pad token | Natural for autoregressive tasks | Less suitable for classification |

**How [CLS] Token Pooling Works:**

After the final transformer layer, we have hidden states for all positions:

```python
# After final transformer layer:
# Shape: [batch_size, seq_len, d_model]
hidden_states = [
    [CLS vector],      # Position 0 ← We keep this! ✅
    [char 'g' vector], # Position 1 ← Discarded ❌
    [char 'o' vector], # Position 2 ← Discarded ❌
    [char 'o' vector], # Position 3 ← Discarded ❌
    [char 'g' vector], # Position 4 ← Discarded ❌
    [char 'l' vector], # Position 5 ← Discarded ❌
    [char 'e' vector], # Position 6 ← Discarded ❌
    [char 'c' vector], # Position 7 ← Discarded ❌
    [char 'o' vector], # Position 8 ← Discarded ❌
    [char 'm' vector], # Position 9 ← Discarded ❌
    [<PAD> vectors],   # Position 10-63 ← Discarded ❌
]

# [CLS] Token Pooling: Extract position 0, discard all others
cls_vector = hidden_states[:, 0]  # Shape: [batch_size, d_model]

# Why can we discard the other vectors?
# Because the [CLS] vector already contains information from 
# ALL positions through self-attention!
```

**Key Insight**: Through self-attention in each transformer layer, the `<CLS>` token "looks at" all character positions and learns to aggregate their information. By the final layer, the `<CLS>` vector contains a rich representation of the entire domain, optimized specifically for classification.

#### 6. Classification Head

Final linear layer maps pooled representation to class logits:

```python
classifier = nn.Linear(d_model, num_classes=2)
logits = classifier(pooled)  # [batch, 2]
```

## Design Decisions

### 1. Sequence Length: 64 Characters

**Decision**: `max_len=64`

**Empirical Analysis** (from 16.2M domain dataset):
```python
# Domain length distribution statistics
Dataset size:       16,246,006 domains

Percentiles:
  95th percentile:  25 chars
  99th percentile:  29 chars
  99.9th percentile: 34 chars

Range:
  Minimum:           0 chars
  Maximum:          63 chars
  Mean:           13.6 chars

Coverage at max_len=64:
  100.00% 
```

**Why max_len=64 is optimal**:
- **Zero truncation**: All domains in the dataset fit within 64 characters
- **Computational efficiency**: Keeps sequence length power-of-2 aligned for GPU optimization
- **Memory footprint**: Reasonable for batch processing (can fit 2048+ samples on modern GPUs)
- **Headroom**: Leaves 30+ chars of buffer above 99.9th percentile for edge cases

Alternative analysis for different max_len values:
- `max_len=32` would cover ~96% (truncates 4%)
- `max_len=64` covers 100% ✓

`pixi run python data_analysis.py`

### 2. Model Depth vs. Width Trade-off

| Profile | Layers | Width (d_model) | Params | Use Case |
|---------|--------|-----------------|--------|----------|
| `tiny`  | 4      | 256             | 3.2M   | Quick experiments, edge deployment |
| `small` | 6      | 384             | 10.8M  | Balanced performance |

**Design philosophy**:
- **Depth** (layers): Captures compositional patterns (character → n-gram → substring → domain structure)
- **Width** (d_model): Increases representational capacity per layer

### 3. Training Hyperparameters

```python
# From config.py - chosen via grid search
lr = 3e-4              # AdamW learning rate
weight_decay = 0.01    # L2 regularization
dropout = 0.1          # Prevent co-adaptation
```

**Key choices**:
- **AdamW**: Better than Adam for transformers (decoupled weight decay)
- **Low dropout (0.1)**: Data is clean, model isn't overfitting