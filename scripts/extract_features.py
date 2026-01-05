import argparse
import numpy as np
from repro.features import FeatureConfig, compute_features_for_rows, load_rows
from repro.utils import save_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", default=None, help="LoRA adapter dir; if omitted, uses base model only")
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--ctx_len", type=int, default=768)
    ap.add_argument("--no_uplift", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    rows = load_rows(args.data_jsonl)
    cfg = FeatureConfig(ctx_len=args.ctx_len, use_uplift=(not args.no_uplift and args.lora_dir is not None))
    X, y, ids, feat_names = compute_features_for_rows(
        base_model_id=args.base_model,
        lora_dir=args.lora_dir,
        rows=rows,
        cfg=cfg,
        batch_size=args.batch_size,
    )
    save_npz(args.out_npz, X=X, y=y, ids=np.array(ids), feature_names=np.array(feat_names))
    print(f"Saved features X={X.shape} -> {args.out_npz}")

if __name__ == "__main__":
    main()
