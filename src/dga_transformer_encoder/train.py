"""Train DGA classifier using HuggingFace Trainer.

This replaces the custom training loop in train.py with HF Trainer, which provides:
- Automatic checkpointing and resumption
- Built-in W&B logging
- Mixed precision training
- Gradient accumulation
- Early stopping
- Distributed training support

Usage:
    pixi run python -m dga_transformer_encoder.train --help
    pixi run python -m dga_transformer_encoder.train --size tiny --epochs 20
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate
import numpy as np
import wandb

from .model import build_model
from .data import prepare_datasets, create_data_collator
from .config import PROFILES


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
)


def compute_metrics(eval_pred):
    """Compute F1, precision, recall, accuracy using HF evaluate library.
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer.evaluate()
    """
    # Load metrics from HF evaluate library
    metric_f1 = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_accuracy = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Compute all metrics
    results = {
        "f1_macro": metric_f1.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"],
        "f1_binary": metric_f1.compute(
            predictions=predictions, references=labels, average="binary"
        )["f1"],
        "precision": metric_precision.compute(
            predictions=predictions, references=labels, average="binary"
        )["precision"],
        "recall": metric_recall.compute(
            predictions=predictions, references=labels, average="binary"
        )["recall"],
        "accuracy": metric_accuracy.compute(
            predictions=predictions, references=labels
        )["accuracy"],
    }

    return results



def train(
    # Model config
    size: str = "tiny",
    # Data
    data_dir: Path = Path("data"),
    # Training hyperparameters
    epochs: int = 5,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    warmup_steps: int = 500,
    # Checkpointing
    output_dir: Path = Path("checkpoints"),
    save_steps: int = 500,
    save_total_limit: int = 3,
    # Evaluation
    eval_steps: int = 500,
    early_stopping_patience: int = 3,
    # Logging
    logging_steps: int = 1000,
    use_wandb: bool = True,
    wandb_project: str = "dga-transformer",
    wandb_run_name: Optional[str] = None,
    # System
    seed: int = 42,
    num_workers: int = 4 * torch.cuda.device_count(),
    # Resume
    resume_from_checkpoint: Optional[Path] = None,
) -> None:
    """Train DGA classifier with HuggingFace Trainer + W&B logging."""
    # Set seed for reproducibility
    set_seed(seed)
    

    # Get profile defaults
    profile = PROFILES[size]
    lr = learning_rate if learning_rate is not None else profile.lr
    wd = weight_decay if weight_decay is not None else profile.weight_decay

    # Initialize W&B if enabled
    if use_wandb:
        run_name = wandb_run_name or f"{size}-lr{lr}-bs{batch_size}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "size": size,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "weight_decay": wd,
            },
        )

    # Load datasets
    print(f"Loading datasets from {data_dir}")
    datasets = prepare_datasets(data_dir)
    print(f"   Train: {len(datasets['train'])} examples")
    print(f"   Val:   {len(datasets['validation'])} examples")
    print(f"   Test:  {len(datasets['test'])} examples")

    # Build model
    print(f"Building {size} model")
    model = build_model(size)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Create data collator
    data_collator = create_data_collator()

    # Training arguments with W&B integration
    training_args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        # Training
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        # Optimization
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=logging_steps,
        report_to=["wandb"] if use_wandb else ["none"],
        run_name=wandb_run_name or f"{size}-{lr}",
        # System
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        seed=seed,
        # Performance
        remove_unused_columns=False,  # Keep 'domain' for debugging
    )

    # Initialize Trainer (replaces entire custom training loop!)
    print("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        ],
    )

    # Check if we're resuming from a checkpoint
    resume_path = resume_from_checkpoint
    if resume_path is None and output_dir.exists():
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_path = max(
                checkpoints, key=lambda p: int(p.name.split("-")[1])
            )

    # Only evaluate baseline if starting fresh (not resuming)
    if resume_path is None:
        print("\nEvaluating baseline (untrained model)")
        baseline_results = trainer.evaluate(
            datasets["validation"], metric_key_prefix="baseline"
        )
        print(
            f"   Baseline F1 (macro): {baseline_results['baseline_f1_macro']:.4f}"
        )
        print(
            f"   Baseline F1 (binary): {baseline_results['baseline_f1_binary']:.4f}"
        )
        print(
            f"   Baseline Accuracy: {baseline_results['baseline_accuracy']:.4f}"
        )

        # Log baseline to W&B
        if use_wandb:
            wandb.log(
                {
                    "baseline/f1_macro": baseline_results["baseline_f1_macro"],
                    "baseline/f1_binary": baseline_results[
                        "baseline_f1_binary"
                    ],
                    "baseline/accuracy": baseline_results["baseline_accuracy"],
                    "baseline/precision": baseline_results[
                        "baseline_precision"
                    ],
                    "baseline/recall": baseline_results["baseline_recall"],
                }
            )
    else:
        print(f"\nResuming from checkpoint: {resume_path}")
        print("   (skipping baseline evaluation)")

    # Train (automatic checkpointing, resumption, W&B logging!)
    print("\nStarting training")
    print(f"   Total steps: {len(datasets['train']) // batch_size * epochs}")

    trainer.train(
        resume_from_checkpoint=str(resume_path) if resume_path else None
    )

    # Evaluate on test set
    print("\nEvaluating on test set")
    test_results = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    print(f"   Test F1 (macro): {test_results['test_f1_macro']:.4f}")
    print(f"   Test F1 (binary): {test_results['test_f1_binary']:.4f}")
    print(f"   Test Accuracy: {test_results['test_accuracy']:.4f}")

    # Save final model
    final_path = output_dir / "final"
    print(f"\nSaving final model to {final_path}")
    trainer.save_model(str(final_path))

    # Log test results to W&B
    if use_wandb:
        wandb.log(test_results)
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    import typer

    typer.run(train)
