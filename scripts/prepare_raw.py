import argparse
import os
from repro.data import prepare_raw
from repro.utils import write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "humaneval"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = prepare_raw(args.dataset)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
