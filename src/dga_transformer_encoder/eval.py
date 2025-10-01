"""Evaluate DGA classifier using HuggingFace Trainer.

This replaces the custom evaluation loop in eval.py with HF Train    print("\nEvaluation complete!")


def predict(
    checkpoint: Path,dict(),
which provides automatic batching and metric computation.

Usage:
    pixi run python -m dga_transformer_encoder.eval ./checkpoints/final --split test
    pixi run eval-test         # Evaluate on test set
    pixi run eval-val          # Evaluate on validation set
    pixi run eval ./checkpoints/final --split test  # Custom eval
    # Prediction
    pixi run predict ./checkpoints/final google.com xkwjfpqmz.org
"""

from __future__ import annotations

from pathlib import Path

import typer
import torch
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .model import DGAEncoderForSequenceClassification
from .data import prepare_datasets, create_data_collator


def eval_model(
    checkpoint: Path,
    data_dir: Path = Path("data"),
    split: str = "test",
    batch_size: int = 256,
    detailed: bool = False,
) -> None:
    """Evaluate a trained DGA classifier.

    This replaces the custom evaluation loop with HF Trainer, eliminating
    manual batching and metric computation.
    """
    print(f"Evaluating model from {checkpoint}")

    # Load datasets
    datasets = prepare_datasets(data_dir)
    eval_dataset = datasets[split]
    print(f"   Dataset: {split} ({len(eval_dataset)} examples)")

    # Load model
    print("Loading model")
    model = DGAEncoderForSequenceClassification.from_pretrained(checkpoint)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create Trainer for evaluation (no training)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
        ),
        data_collator=create_data_collator(),
    )

    # Predict (automatic batching!)
    print("Running inference")
    predictions_output = trainer.predict(eval_dataset)

    # Extract predictions and labels
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=-1)

    # Compute metrics using evaluate library
    print("\nResults:")

    metric_f1_macro = evaluate.load("f1")
    metric_f1_binary = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_accuracy = evaluate.load("accuracy")

    f1_macro = metric_f1_macro.compute(
        predictions=preds, references=labels, average="macro"
    )["f1"]
    f1_binary = metric_f1_binary.compute(
        predictions=preds, references=labels, average="binary"
    )["f1"]
    precision = metric_precision.compute(
        predictions=preds, references=labels, average="binary"
    )["precision"]
    recall = metric_recall.compute(
        predictions=preds, references=labels, average="binary"
    )["recall"]
    accuracy = metric_accuracy.compute(predictions=preds, references=labels)[
        "accuracy"
    ]

    print(f"   Accuracy:       {accuracy:.4f}")
    print(f"   F1 (macro):     {f1_macro:.4f}")
    print(f"   F1 (binary):    {f1_binary:.4f}")
    print(f"   Precision:      {precision:.4f}")
    print(f"   Recall:         {recall:.4f}")

    # Detailed classification report
    if detailed:
        print("\nClassification Report:")
        report = classification_report(
            labels, preds, target_names=["Normal", "DGA"]
        )
        print(report)

        print("\nConfusion Matrix:")
        cm = confusion_matrix(labels, preds)
        print("                Predicted")
        print("              Normal  DGA")
        print(f"Actual Normal {cm[0][0]:6d} {cm[0][1]:6d}")
        print(f"       DGA    {cm[1][0]:6d} {cm[1][1]:6d}")

    print("\nEvaluation complete!")


def predict(
    checkpoint: Path,
    domains: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Predict whether domains are DGA or normal.

    Example:
        python -m dga_transformer_encoder.eval predict ./checkpoints/final google.com xkwjfpqmz.org
    """
    from .charset import encode_domain

    print(f"Loading model from {checkpoint}")
    model = DGAEncoderForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    print(f"\nClassifying {len(domains)} domain(s):\n")

    # Encode domains
    input_ids = torch.tensor(
        [encode_domain(d) for d in domains], dtype=torch.long, device=device
    )

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    # Display results
    for domain, pred, prob in zip(domains, predictions, probs):
        label = "DGA" if pred == 1 else "Normal"
        confidence = prob[pred].item()
        marker = "[!]" if pred == 1 else "[+]"
        print(
            f"{marker} {domain:30s} -> {label:6s} (confidence: {confidence:.2%})"
        )


if __name__ == "__main__":
    typer.run(predict)
