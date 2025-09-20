import os, json, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from charset import VOCAB_SIZE
from model import build_dga_encoder, encoder_from_config

class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.X, self.y = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.X.append(j["ids"])  # pre-encoded ids
                self.y.append(j["label"])  # 0/1
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    te = JsonlDataset(os.path.join(args.data, "test.jsonl"))
    dl_te = DataLoader(te, batch_size=8192)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Prefer exact reconstruction from saved config; otherwise fall back to size preset
    if "config" in ckpt:
        model = encoder_from_config(ckpt["config"], vocab_size=VOCAB_SIZE)
    elif "size" in ckpt:
        model = build_dga_encoder(ckpt["size"], vocab_size=VOCAB_SIZE)
    else:
        raise ValueError("Checkpoint lacks both 'config' and 'size'")

    model.load_state_dict(ckpt["state_dict"])  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds, gts = [], []
    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(device, non_blocking=True)
            p = model(x).argmax(1).cpu().tolist()
            preds += p; gts += y.tolist()

    print("macro-F1:", f1_score(gts, preds, average="macro"))
    print("confusion matrix:\n", confusion_matrix(gts, preds))
    print(classification_report(gts, preds, digits=4))
