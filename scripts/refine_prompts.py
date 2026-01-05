import argparse
import os
from tqdm import tqdm
from repro.utils import read_jsonl, write_jsonl
from repro.refine import HFRefiner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "humaneval"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refiner_model", required=True, help="HF model id for local refinement")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    refiner = HFRefiner(model_id=args.refiner_model)

    out_rows = []
    for r in tqdm(rows, desc="refine"):
        x_ref = refiner.generate_one(args.task, r["x_raw"])
        out_rows.append({"id": r["id"], "x_ref": x_ref})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_jsonl(args.out, out_rows)
    print(f"Wrote refined prompts -> {args.out}")

if __name__ == "__main__":
    main()
