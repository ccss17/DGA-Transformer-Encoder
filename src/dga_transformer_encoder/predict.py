"""
pixi run predict ./checkpoints/final google.com xkwjfpqmz.org
python -m dga_transformer_encoder.predict ./checkpoints/final google.com facebook.com malicious123.xyz
"""

from pathlib import Path

import torch

from .model import DGAEncoderForSequenceClassification
from .charset import encode_domain


def predict(
    checkpoint: Path,
    domains: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Predict whether domains are DGA or normal.

    checkpoint: Path to the trained model checkpoint
    domains: List of domain names to classify
    device: Device to run inference on (cuda or cpu)
    """
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
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Display results
    for domain, pred, prob in zip(domains, predictions, probs):
        label = "DGA" if pred == 1 else "Normal"
        confidence = prob[pred].item()
        marker = "[!]" if pred == 1 else "[+]"
        print(
            f"{marker} {domain:30s} -> {label:6s} (confidence: {confidence:.2%})"
        )


if __name__ == "__main__":
    import typer
    typer.run(predict)
