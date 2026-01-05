import argparse
import json
import numpy as np
import torch
from repro.utils import load_npz
from repro.attacker import load_models
from repro.eval import compute_metrics, plot_roc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", required=True)
    ap.add_argument("--attacker_dir", required=True)
    ap.add_argument("--out_json", required=True)
    # ap.add_argument("--plot_png", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = load_npz(args.features_npz)
    X, y = data["X"], data["y"]

    encoder, head, artifacts = load_models(args.attacker_dir, in_dim=X.shape[1], hidden=args.hidden, emb_dim=args.emb_dim, device=args.device)
    X_std = artifacts.standardizer.transform(X)
    with torch.no_grad():
        u = encoder(torch.tensor(X_std, dtype=torch.float32, device=args.device))
        logits = head(u)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

    metrics = compute_metrics(y, scores)
    # plot_roc(y, scores, args.plot_png)

    out = {"metrics": metrics, "n": int(len(y))}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
