
# DGA Transformer Encoder

A transformer-based neural network for detecting Domain Generation Algorithm (DGA) domains used in malware command-and-control infrastructure.  **96.78% F1 Score** on test set (tiny model 3.2M parameters)

**Domain Generation Algorithms (DGAs)** are techniques used by malware to generate large numbers of pseudo-random domain names for C2 (command-and-control) communication. This makes it harder for security systems to block malicious domains through blacklisting.

**Examples:**
- **Legitimate domains**: `google.com`, `github.com`, `stackoverflow.com`
- **DGA domains**: `xjkd8f2h.com`, `qwfp93nx.net`, `h4fk29fd.org`

## Installation

### Setup with Pixi

by [Pixi](https://pixi.sh/):

```bash
pixi install
```

## Usage

1. Prepare Data

```bash
mkdir -p data/raw && cd data/raw
curl -L -o dga-training-data-encoded.json.gz \
  https://github.com/ExtraHop/DGA-Detection-Training-Dataset/raw/refs/heads/main/dga-training-data-encoded.json.gz
cd ../..
pixi run prepare-data
```

- Reads `data/raw/dga-training-data-encoded.json.gz`
- Tokenizes domains into character IDs
- Splits: 80% train, 10% validation, 10% test
- Saves to `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`

2. Train Model

Train with default settings (batch_size=2048):

```bash
pixi run train
```








## Results

### Tiny Model (3.2M params, 5 epochs)

**Baseline Evaluation**

```
Baseline F1 (macro): 0.3333  (random guessing)
Baseline Accuracy:   0.5000
```

**Test Set Performance**:
```
Accuracy:  96.78%
Precision: 97.02%
Recall:    96.54%
F1 (macro):  96.78%
F1 (binary): 96.78%
```


- Converged in ~2.5 epochs (early stopping at epoch 2.4)
- Train time: ~12 minutes total (RTX 4060 Ti, batch_size=2048)
- GPU memory: 1.2 GB



# License

MIT
