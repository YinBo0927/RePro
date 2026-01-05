import argparse
import os
from repro.utils import load_npz
from repro.attacker import train_attacker
from repro.eval import choose_threshold_at_fpr
import json
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow_train_npz", required=True)
    ap.add_argument("--shadow_val_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tr = load_npz(args.shadow_train_npz)
    va = load_npz(args.shadow_val_npz)

    Xtr, ytr = tr["X"], tr["y"]
    Xva, yva = va["X"], va["y"]
    feature_names = [str(x) for x in tr.get("feature_names", [])]

    train_attacker(
        X_train=Xtr, y_train=ytr,
        X_val=Xva, y_val=yva,
        out_dir=args.out_dir,
        hidden=args.hidden, emb_dim=args.emb_dim,
        batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, temperature=args.temperature,
        device=args.device,
        feature_names=feature_names if feature_names else None,
    )

    # Compute and save threshold for 1% FPR on shadow val using the trained head.
    from repro.attacker import load_models
    from repro.utils import Standardizer
    from repro.eval import compute_metrics

    encoder, head, artifacts = load_models(args.out_dir, in_dim=Xva.shape[1], hidden=args.hidden, emb_dim=args.emb_dim, device=args.device)
    Xva_std = artifacts.standardizer.transform(Xva)
    with torch.no_grad():
        u = encoder(torch.tensor(Xva_std, dtype=torch.float32, device=args.device))
        logits = head(u)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

    thr_1pct = choose_threshold_at_fpr(yva, scores, target_fpr=0.01)
    metrics = compute_metrics(yva, scores)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"thr_at_1pct_fpr": thr_1pct, "shadow_val_metrics": metrics}, f, indent=2)
    print("Saved thresholds.json")

if __name__ == "__main__":
    main()
