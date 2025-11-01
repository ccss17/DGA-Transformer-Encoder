
trained on vast.ai with [vastai_quick_setup](https://github.com/ccss17/vastai_quick_setup)

- wandb: https://wandb.ai/cccsss17-xxx/dga-transformer
- HF spaces(live demo): https://huggingface.co/spaces/ccss17/dga-detector
- HF models: https://huggingface.co/ccss17/dga-transformer-encoder

# DGA Detection Transformer Encoder

* **Benign domain**: `google.com`, **DGA domain**: `xjkd8f2h.com`

## Model Design


|           Parameter | Value |
| ------------------: | :---- |
|              Layers | 4     |
|    Hidden Dimension | 256   |
|     Attention Heads | 8     |
|      FFN Multiplier | 4     |
|             Dropout | 0.1   |
|       Learning Rate | 3e-4  |
|        Weight Decay | 0.01  |
| Max Sequence Length | 64    |

## Project Design

**Dataset construction**:

* **Source**: GitHub: [ExtraHop/DGA-Detection-Training-Dataset](https://github.com/ExtraHop/DGA-Detection-Training-Dataset)
* **Preprocessing**: Tokenize domains at character level and convert to character ID sequences
* **Split**: Train 80% / Validation 10% / Test 10%

**Training setup**:

* **batch**: 512
* **epoch**: 5 (early stopping `patience=3`)
* **training time**: about 12 minutes (RTX 4060 Ti) — `Train Loggin Wandb`: [https://wandb.ai/cccsss17-xxx/dga-transformer/runs/hn10av19?nw=nwusercccsss17](https://wandb.ai/cccsss17-xxx/dga-transformer/runs/hn10av19?nw=nwusercccsss17)
* **convergence**: early stopped at 2.4 epochs

## Results

Hugging Face model: [ccss17/dga-transformer-encoder](https://huggingface.co/ccss17/dga-transformer-encoder)

Hugging Face demo (Spaces): [ccss17/dga-detector Demo Spaces](https://huggingface.co/spaces/ccss17/dga-detector)

| Metric       | Baseline | After training |
| ------------ | -------: | -------------: |
| **Accuracy** |   50.00% |         96.78% |
| **F1**       |   33.00% |         96.78% |


## Dev

```bash
pixi install

mkdir -p data/raw && cd data/raw
curl -L -o dga-training-data-encoded.json.gz \
  https://github.com/ExtraHop/DGA-Detection-Training-Dataset/raw/refs/heads/main/dga-training-data-encoded.json.gz
cd ../..
pixi run prepare-data

pixi run train
```
# License

MIT
