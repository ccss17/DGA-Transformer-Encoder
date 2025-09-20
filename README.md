```shell
#!/usr/bin/env bash
###############################################
# 0) Create Env & Download dataset
###############################################

uv sync

# Source (GitHub): https://github.com/ExtraHop/DGA-Detection-Training-Dataset
mkdir -p data/raw
cd data/raw

curl -L -o dga-training-data-encoded.json.gz \
  https://github.com/ExtraHop/DGA-Detection-Training-Dataset/raw/refs/heads/main/dga-training-data-encoded.json.gz

cd ../..

###############################################
# 1) Prepare splits (train/val/test) from the raw file
###############################################
# --per-class controls how many benign and dga examples you sample per class.
# Feel free to raise it if you have a bigger GPU or more time.

uv run src/dga_transformer_encoder/prepare_data.py \
  --input data/raw/dga-training-data-encoded.json.gz \
  --per-class 250000 \
  --out data

# This generates: data/train.jsonl, data/val.jsonl, data/test.jsonl

###############################################
# 2) Train (Tiny/Small/Medium/Heavy). Resumable by design.
# | size   | params(≈) | typical batch | min VRAM\* | recommended |
# | ------ | --------: | ------------- | ---------- | ----------- |
# | tiny   |    \~3.2M | 2048          | 8 GB       | 10–12 GB    |
# | small  |   \~10.8M | 1024          | 12 GB      | 16 GB       |
# | medium |   \~25.6M | 1024          | 16 GB      | 24 GB       |
# | heavy  |   \~85.2M | 512–1024      | 24 GB\*\*  | 40–48 GB    |
###############################################
# Tiny (default quick run)
uv run src/dga_transformer_encoder/train.py \
  --size tiny \
  --data data \
  --epochs 5 \
  --batch-size 2048 \
  --save-dir ckpt_tiny \
  --save-best \
  --ckpt-interval-steps 200 \
  --ckpt-interval-seconds 180

# Resume after a preemption (last checkpoint is used by default)
python train.py \
  --size tiny \
  --data data \
  --save-dir ckpt_tiny \
  --resume

# Small (heavier, still fast on L40S/A100/H100)
python train.py \
  --size small \
  --data data \
  --epochs 6 \
  --batch-size 1024 \
  --save-dir ckpt_small \
  --save-best

###############################################
# 3) Evaluate on the held-out test set
###############################################
python eval.py \
  --data data \
  --ckpt ckpt_tiny/best.pt

###############################################
# 4) (Optional) Demo with the trained checkpoint
###############################################
python demo.py \
  --ckpt ckpt_tiny/best.pt

###############################################
# 5) (Optional) Weights & Biases logging + artifact upload
###############################################
# Use train_wandb.py instead of train.py if you want W&B.
# Provide a run id to resume across preemptions; enable artifact upload for best.pt.

# Example run
python train_wandb.py \
  --size small \
  --data data \
  --epochs 8 \
  --batch-size 1024 \
  --save-dir ckpt_small \
  --save-best \
  --ckpt-interval-steps 200 \
  --ckpt-interval-seconds 180 \
  --wandb \
  --wandb-project dga-transformer \
  --wandb-id run_abc123 \
  --tags tiny,cu128,fa2 \
  --wandb-artifact

# Resume the same W&B run after a preemption
WANDB_RESUME=allow python train_wandb.py \
  --size small \
  --data data \
  --save-dir ckpt_small \
  --wandb \
  --wandb-project dga-transformer \
  --wandb-id run_abc123 \
  --resume

###############################################
# 6) (Optional) Switch backends via uv extras (if your repo uses them)
###############################################
# Local (CUDA 11.8) example:
#   uv sync --extra cu118
# Vast.ai (CUDA 12.8) example:
#   uv sync --extra cu128
# CPU-only smoke test:
#   uv sync --extra cpu
```


```shell
uv sync

# 1) Prepare dataset (balanced sample from ExtraHop JSONL.GZ)
#    --input: path to ExtraHop file (JSON lines, gzipped) with fields {"domain": ..., "threat": "benign"|"dga"}
#    --per-class: number of domains per class to sample (e.g., 250k for quick training)
python prepare_data.py --input data/extrahop_dga.jsonl.gz --per-class 250000 --out data

# This will create data/train.jsonl, data/val.jsonl, data/test.jsonl

# 2) Train (Tiny profile by default)
python train.py --profile tiny --data data --epochs 5 --batch-size 2048 --save-dir ckpt_tiny

# 3) Evaluate
python eval.py --data data --ckpt ckpt_tiny/best.pt

# 4) Demo (will load ckpt + tokenizer)
python demo.py --ckpt ckpt_tiny/best.pt

# (Option) Train Small profile
python train.py --profile small --data data --epochs 6 --batch-size 1024 --save-dir ckpt_small
```

> 데이터는 domain(label)만을 입력으로 사용합니다(ExtraHop는 TLD 제외). Demo에서 URL을 입력하면 내부에서 URL→registrable domain(+TLD 제거 시도)→tokenize 과정을 거칩니다.


- Profiles: tiny(~3–4M params) → fastest; small(~10–15M) → slightly higher ceiling. 둘 다 H100/H200에서 AMP+Flash SDPA로 매우 빠르게 수렴합니다.
- Input: ExtraHop 데이터는 도메인(라벨)만, TLD 제외입니다. URL로 입력받을 땐 demo가 자동으로 추출/정규화합니다.
- Metrics: macro-F1 + confusion matrix(출력은 eval.py 참조)
- Ethics: 교육·연구 목적. 실제 차단/탐지 의사결정은 DNS 텔레메트리·등록정보·시간 흐름과 결합하여 수행하십시오.