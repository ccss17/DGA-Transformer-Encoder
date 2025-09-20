import argparse, json, re, urllib.parse

try:
    import tldextract  # optional, for robust registrable-domain parsing
except Exception:
    tldextract = None

import torch
from model import TinyDGAEncoder
from charset import VOCAB_SIZE, encode_domain
import gradio as gr

LABELS = ["benign", "dga"]

def url_to_domain_label(s: str) -> str:
    """Extract registrable-domain label (no TLD) from URL or plain domain string.
    If tldextract is available, use it; otherwise use a heuristic fallback.
    Example: https://sub.example.co.uk/path -> "example"
    """
    s = (s or "").strip()
    host = s
    try:
        host = urllib.parse.urlparse(s).hostname or s
    except Exception:
        pass
    host = host.strip().lower()
    if tldextract:
        ext = tldextract.extract(host)
        return ext.domain  # registrable label without TLD
    # Fallback heuristic: take the second-level label before the last dot
    parts = host.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return host

class Predictor:
    def __init__(self, ckpt):
        ckpt = torch.load(ckpt, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.model = TinyDGAEncoder(vocab_size=VOCAB_SIZE, max_len=cfg.get("max_len",64),
                                    d_model=cfg.get("d_model",256), nhead=cfg.get("nhead",8),
                                    nlayers=cfg.get("layers",4), ffn_mult=cfg.get("ffn_mult",4))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().cuda()

    @torch.no_grad()
    def __call__(self, text: str):
        d = url_to_domain_label(text)
        ids = torch.tensor([encode_domain(d)], dtype=torch.long).cuda()
        logits = self.model(ids)
        prob = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        top = int(torch.argmax(logits, dim=-1)[0])
        return {LABELS[i]: float(prob[i]) for i in range(2)}, f"domain='{d}' => {LABELS[top]}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    predictor = Predictor(args.ckpt)

    demo = gr.Interface(
        fn=predictor,
        inputs=gr.Textbox(lines=2, label="Enter URL or domain"),
        outputs=[gr.Label(num_top_classes=2), gr.Textbox(label="Decision")],
        title="DGA Domain Classifier (Transformer Encoder)",
        description="This model predicts whether a registrable-domain label (no TLD) is DGA/benign.",
    )
    demo.launch()