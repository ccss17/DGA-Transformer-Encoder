from pathlib import Path

from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .model import DGAEncoderForSequenceClassification
from .data import prepare_datasets, create_data_collator
from .train import compute_metrics


def eval_model(
    checkpoint: Path,
    data_dir: Path = Path("data"),
    split: str = "test",
    batch_size: int = 256,
) -> None:
    """replaces the custom evaluation loop with HF Trainer, eliminating
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
            dataloader_pin_memory=True,
        ),
        data_collator=create_data_collator(),
        compute_metrics=compute_metrics,
    )

    # Evaluate using Trainer (automatic batching and metrics!)
    print("Running evaluation")
    
    # Get predictions and metrics in one pass
    predictions_output = trainer.predict(eval_dataset, metric_key_prefix=split)
    results = predictions_output.metrics
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=-1)
    
    # Print results
    print("\nResults:")
    print(f"   {results}")

    # Detailed classification report
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


if __name__ == "__main__":
    import typer
    typer.run(eval_model)
