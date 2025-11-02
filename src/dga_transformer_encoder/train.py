"""
replaces custom training loop with HF Trainer:
- Automatic checkpointing and resumption
- Built-in W&B logging
- Mixed precision training
- Gradient accumulation
- Early stopping
- Distributed training support

pixi run python -m dga_transformer_encoder.train --help
pixi run python -m dga_transformer_encoder.train --size tiny --epochs 20
"""

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


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    return max(checkpoints, key=lambda p: int(p.name.split("-")[1]))


def compute_metrics(eval_pred):
    """eval_pred: Tuple of (predictions, labels) from Trainer.evaluate()
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return metrics.compute(predictions=predictions, references=labels)


def train(
    # Model config
    size: str = "tiny",
    # Data
    data_dir: str = "data",
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
):
    set_seed(seed)

    profile = PROFILES[size]
    lr = learning_rate if learning_rate is not None else profile.lr
    wd = weight_decay if weight_decay is not None else profile.weight_decay

    # Determine run name for W&B and Trainer
    run_name = wandb_run_name or f"{size}-lr{lr}-bs{batch_size}"

    # Initialize W&B if enabled
    if use_wandb:
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
        metric_for_best_model="f1",
        greater_is_better=True,
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=logging_steps,
        report_to=["wandb"] if use_wandb else ["none"],
        run_name=run_name,
        # System
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        seed=seed,
        # Performance
        remove_unused_columns=False,  # Keep 'domain' for debugging
    )

    # Initialize Trainer (replaces entire custom training loop)
    print("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=create_data_collator(),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        ],
    )

    # auto-detect a checkpoint if the user didn't explicitly specify one
    if resume_from_checkpoint is None:
        resume_from_checkpoint = find_latest_checkpoint(output_dir)

    # Only evaluate baseline if starting fresh (not resuming)
    if resume_from_checkpoint is None:
        print("\nEvaluating baseline (untrained model)")
        baseline_results = trainer.evaluate(
            datasets["validation"], metric_key_prefix="baseline"
        )
        print(f"   {baseline_results}")

        # Log baseline to W&B
        if use_wandb:
            wandb.log(baseline_results)
    else:
        print(f"\nResuming from checkpoint: {resume_from_checkpoint}")
        print("   (skipping baseline evaluation)")

    # Train (automatic checkpointing, resumption, W&B logging)
    print("\nStarting training")
    print(f"   Total steps: {len(datasets['train']) // batch_size * epochs}")
    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )

    # Evaluate on test set
    print("\nEvaluating on test set")
    test_results = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    print(f"   {test_results}")

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
