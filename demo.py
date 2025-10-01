"""Gradio demo for DGA domain classifier.

This interactive web app lets users test the trained model on domain names.
It provides predictions with confidence scores and visual feedback.
"""

import torch
import gradio as gr
from pathlib import Path

from src.dga_transformer_encoder.model import (
    DGAEncoderForSequenceClassification,
)
from src.dga_transformer_encoder.charset import encode_domain


# Load model at startup
MODEL_PATH = Path("checkpoints/final")
print(f"Loading model from {MODEL_PATH}...")
model = DGAEncoderForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")


def predict_domain(domain: str):
    """Classify a domain as legitimate or DGA-generated.

    Args:
        domain: Domain name to classify (e.g., "google.com", "xjkd8f2h.com")

    Returns:
        tuple: (prediction_label, confidence_score, html_output)
    """
    if not domain or not domain.strip():
        return "⚠️ Invalid Input", "", "Please enter a domain name."

    domain = domain.strip().lower()

    # Encode domain to token IDs
    input_ids = torch.tensor(
        [encode_domain(domain, max_len=64)], device=device
    )
    attention_mask = (input_ids != 0).long()

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()

    # Format output
    label_names = ["✅ Legitimate", "🚨 DGA (Malicious)"]
    prediction = label_names[pred_class]
    confidence_pct = f"{confidence * 100:.2f}%"

    # Create detailed HTML output
    legit_prob = probs[0, 0].item()
    dga_prob = probs[0, 1].item()

    html_output = f"""
    <div style="padding: 20px; border-radius: 10px; background: {"#d4edda" if pred_class == 0 else "#f8d7da"};">
        <h2 style="margin-top: 0; color: {"#155724" if pred_class == 0 else "#721c24"};">
            {prediction}
        </h2>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>Domain:</strong> <code>{domain}</code>
        </p>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>Confidence:</strong> {confidence_pct}
        </p>
        <hr style="margin: 15px 0; border: none; border-top: 1px solid #ccc;">
        <h3>Probability Breakdown:</h3>
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>✅ Legitimate:</span>
                <strong>{legit_prob * 100:.2f}%</strong>
            </div>
            <div style="background: #e9ecef; border-radius: 5px; overflow: hidden; height: 20px;">
                <div style="background: #28a745; height: 100%; width: {legit_prob * 100}%;"></div>
            </div>
        </div>
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>🚨 DGA (Malicious):</span>
                <strong>{dga_prob * 100:.2f}%</strong>
            </div>
            <div style="background: #e9ecef; border-radius: 5px; overflow: hidden; height: 20px;">
                <div style="background: #dc3545; height: 100%; width: {dga_prob * 100}%;"></div>
            </div>
        </div>
    </div>
    """

    return prediction, confidence_pct, html_output


def predict_batch(domains_text: str):
    """Classify multiple domains at once.

    Args:
        domains_text: Newline-separated list of domains

    Returns:
        str: Formatted results table
    """
    if not domains_text or not domains_text.strip():
        return "Please enter one or more domain names (one per line)."

    domains = [
        d.strip() for d in domains_text.strip().split("\n") if d.strip()
    ]

    if not domains:
        return "No valid domains provided."

    results = []
    results.append("| Domain | Prediction | Confidence |")
    results.append("|--------|------------|------------|")

    for domain in domains:
        domain = domain.lower()

        # Encode domain
        input_ids = torch.tensor(
            [encode_domain(domain, max_len=64)], device=device
        )
        attention_mask = (input_ids != 0).long()

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()

        label = "Legitimate ✅" if pred_class == 0 else "DGA 🚨"
        confidence_pct = f"{confidence * 100:.1f}%"

        results.append(f"| `{domain}` | {label} | {confidence_pct} |")

    return "\n".join(results)


# Example domains for quick testing
EXAMPLES = [
    ["google.com"],
    ["github.com"],
    ["stackoverflow.com"],
    ["xjkd8f2h.com"],
    ["qwfp93nx.net"],
    ["h4fk29fd.org"],
    ["facebook.com"],
    ["fjdkslajf.com"],
]

BATCH_EXAMPLES = [
    "google.com\ngithub.com\nstackoverflow.com",
    "xjkd8f2h.com\nqwfp93nx.net\nh4fk29fd.org",
    "amazon.com\nfacebook.com\ntwitter.com\nmicrosoft.com",
]


# Create Gradio interface
with gr.Blocks(title="DGA Domain Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 DGA Domain Classifier
    
    **Detect malicious domains generated by Domain Generation Algorithms (DGAs)**
    
    This model uses a transformer-based neural network to classify domains as either:
    - ✅ **Legitimate**: Normal, human-registered domains
    - 🚨 **DGA (Malicious)**: Algorithmically-generated domains used by malware for C2 communication
    
    ---
    
    ### Model Details
    - **Architecture**: Custom Transformer Encoder (4 layers, 256 dim)
    - **Parameters**: 3.2M
    - **Accuracy**: 96.78% F1 score on test set
    - **Inference Speed**: <1ms per domain
    
    ---
    """)

    with gr.Tabs():
        # Tab 1: Single domain prediction
        with gr.Tab("Single Domain"):
            gr.Markdown("### Test a single domain name")

            with gr.Row():
                with gr.Column(scale=2):
                    domain_input = gr.Textbox(
                        label="Enter Domain Name",
                        placeholder="e.g., google.com, xjkd8f2h.com",
                        lines=1,
                    )
                    predict_btn = gr.Button(
                        "🔍 Classify Domain", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    prediction_output = gr.Textbox(label="Prediction", lines=1)
                    confidence_output = gr.Textbox(label="Confidence", lines=1)

            html_output = gr.HTML(label="Detailed Results")

            predict_btn.click(
                fn=predict_domain,
                inputs=domain_input,
                outputs=[prediction_output, confidence_output, html_output],
            )

            gr.Markdown("### Try these examples:")
            gr.Examples(
                examples=EXAMPLES,
                inputs=domain_input,
                outputs=[prediction_output, confidence_output, html_output],
                fn=predict_domain,
                cache_examples=False,
            )

        # Tab 2: Batch prediction
        with gr.Tab("Batch Prediction"):
            gr.Markdown("### Classify multiple domains at once (one per line)")

            batch_input = gr.Textbox(
                label="Enter Domains (one per line)",
                placeholder="google.com\ngithub.com\nxjkd8f2h.com",
                lines=8,
            )
            batch_btn = gr.Button(
                "🔍 Classify All Domains", variant="primary", size="lg"
            )
            batch_output = gr.Markdown(label="Results")

            batch_btn.click(
                fn=predict_batch,
                inputs=batch_input,
                outputs=batch_output,
            )

            gr.Markdown("### Try these examples:")
            gr.Examples(
                examples=BATCH_EXAMPLES,
                inputs=batch_input,
                outputs=batch_output,
                fn=predict_batch,
                cache_examples=False,
            )

        # Tab 3: About
        with gr.Tab("About"):
            gr.Markdown("""
            ## 📚 What are DGAs?
            
            **Domain Generation Algorithms (DGAs)** are techniques used by malware to generate 
            large numbers of pseudo-random domain names for C2 (command-and-control) communication.
            
            ### Why DGAs are dangerous:
            - **Evasion**: Traditional blacklists can't keep up with thousands of generated domains
            - **Resilience**: Even if some domains are blocked, malware can try others
            - **Stealth**: DGA domains look random, making detection challenging
            
            ### How this model works:
            
            1. **Character-level tokenization**: Breaks domain into individual characters
            2. **Transformer encoder**: Learns patterns in character sequences
            3. **Self-attention**: Detects unusual character combinations (e.g., `xqz`, `fgh`)
            4. **Classification**: Predicts if domain is legitimate or DGA-generated
            
            ### Key Features:
            - **High accuracy**: 96.78% F1 score on test set
            - **Fast inference**: <1ms per domain (GPU) or ~10ms (CPU)
            - **Lightweight**: Only 3.2M parameters
            - **Production-ready**: Trained on real-world malware domains
            
            ### Examples:
            
            **Legitimate domains** (structured, pronounceable):
            - `google.com`, `github.com`, `stackoverflow.com`
            - `api-docs.company.com`, `cdn-assets.example.org`
            
            **DGA domains** (random, unpronounceable):
            - `xjkd8f2h.com`, `qwfp93nx.net`, `h4fk29fd.org`
            - `kdjf92jd.info`, `zmxbv73k.biz`
            
            ---
            
            ### Technical Details:
            - **Model**: Custom Transformer Encoder
            - **Training data**: ExtraHop DGA dataset
            - **Framework**: PyTorch + HuggingFace Transformers
            - **Experiment tracking**: Weights & Biases
            
            ### Links:
            - [GitHub Repository](https://github.com/ccss17)
            - [ExtraHop DGA Dataset](https://github.com/extrahop/dga-training-data)
            
            ---
            
            **Built with ❤️ using PyTorch, HuggingFace, and Gradio**
            """)

    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>⚠️ <strong>Disclaimer</strong>: This model is for educational and research purposes. 
        Always use multiple detection methods in production security systems.</p>
        <p>Model accuracy: 96.78% | False positive rate: ~3%</p>
    </div>
    """)


if __name__ == "__main__":
    # Launch the demo
    print("\n" + "=" * 60)
    print("🚀 Launching DGA Domain Classifier Demo")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Parameters: 3.2M")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="127.0.0.1",  # Bind to all interfaces for SSH tunnel access
        server_port=8080,
        share=False,
        show_error=True,
    )
