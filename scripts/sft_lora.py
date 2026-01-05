import argparse
from repro.sft import train_lora_sft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ctx_len", type=int, default=768)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    args = ap.parse_args()

    train_lora_sft(
        base_model=args.base_model,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        out_dir=args.out_dir,
        ctx_len=args.ctx_len,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

if __name__ == "__main__":
    main()
