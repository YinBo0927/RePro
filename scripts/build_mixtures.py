import argparse
from repro.data import build_mixtures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    build_mixtures(args.raw, args.ref, args.out_dir, rho=args.rho, seed=args.seed)
    print(f"Built mixtures at {args.out_dir}")

if __name__ == "__main__":
    main()
