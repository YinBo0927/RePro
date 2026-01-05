from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .sft import build_inputs_and_labels
from .utils import read_jsonl

@dataclass
class FeatureConfig:
    ctx_len: int = 768
    topk: Sequence[int] = (1, 5, 10)
    nll_quantiles: Sequence[float] = (0.5, 0.9, 0.95)
    use_uplift: bool = True

def _masked_token_stats(logits: torch.Tensor, labels: torch.Tensor, topk: Sequence[int], quantiles: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Compute per-example statistics for masked positions (labels != -100).

    logits: [B, L, V]
    labels: [B, L] with -100 for masked prompt tokens, else token ids
    """
    # Shift for causal LM
    logits_s = logits[:, :-1, :]
    labels_s = labels[:, 1:]
    mask = labels_s != -100  # [B, L-1]

    log_probs = torch.log_softmax(logits_s, dim=-1)
    # Gather logprob of the target token
    target = labels_s.clamp(min=0)
    tgt_lp = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    nll_tok = -tgt_lp  # [B, L-1]

    B = logits.shape[0]
    out: Dict[str, torch.Tensor] = {}

    # NLL mean
    denom = mask.sum(dim=1).clamp(min=1)
    nll_mean = (nll_tok * mask).sum(dim=1) / denom
    out["nll_mean"] = nll_mean

    # NLL quantiles (loop to handle variable lengths)
    qs = []
    for q in quantiles:
        vals = []
        for b in range(B):
            v = nll_tok[b][mask[b]]
            if v.numel() == 0:
                vals.append(torch.tensor(float("nan"), device=logits.device))
            else:
                vals.append(torch.quantile(v, q))
        qs.append(torch.stack(vals, dim=0))
    for i, q in enumerate(quantiles):
        out[f"nll_q{int(q*100):02d}"] = qs[i]

    # Top-k inclusion rates
    max_k = max(topk)
    topk_idx = torch.topk(logits_s, k=max_k, dim=-1).indices  # [B, L-1, max_k]
    for k in topk:
        in_topk = (topk_idx[:, :, :k] == target.unsqueeze(-1)).any(dim=-1)  # [B, L-1]
        rate = (in_topk & mask).sum(dim=1) / denom
        out[f"top{k}_rate"] = rate

    # Margin (top1 - top2) averaged on masked tokens
    top2 = torch.topk(logits_s, k=2, dim=-1).values  # [B, L-1, 2]
    gap = top2[:, :, 0] - top2[:, :, 1]
    out["gap_mean"] = (gap * mask).sum(dim=1) / denom

    # Sum logp of y given x (for pairwise baseline)
    out["logp_sum"] = (tgt_lp * mask).sum(dim=1)

    return out

@torch.no_grad()
def compute_features_for_rows(
    base_model_id: str,
    lora_dir: Optional[str],
    rows: List[Dict[str, Any]],
    cfg: FeatureConfig,
    batch_size: int = 4,
    bf16: bool = True,
    device_map: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return (X, y, ids, feature_names).

    X: [N, d] float32
    y: [N] int64 (z labels)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()

    if lora_dir is not None:
        tuned_model = PeftModel.from_pretrained(base_model, lora_dir).eval()
    else:
        tuned_model = base_model

    # If uplift is requested, we need a separate base model without LoRA attached.
    uplift_base_model = None
    if cfg.use_uplift and lora_dir is not None:
        uplift_base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=dtype, device_map=device_map
        ).eval()

    # Build batched tensors
    feats_all: List[np.ndarray] = []
    labels_all: List[int] = []
    ids_all: List[str] = []

    # Prepare feature name list deterministically
    f_names = ["nll_mean"] + [f"nll_q{int(q*100):02d}" for q in cfg.nll_quantiles] +                   [f"top{k}_rate" for k in cfg.topk] + ["gap_mean", "logp_sum"]
    if cfg.use_uplift:
        f_names += [f"uplift_{name}" for name in ["nll_mean"] + [f"top{k}_rate" for k in cfg.topk] + ["gap_mean"]]

    def collate(batch):
        # pad to max len
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, device=tuned_model.device),
            "labels": torch.tensor(labels, device=tuned_model.device),
            "attention_mask": torch.tensor(attn, device=tuned_model.device),
            "id": [b["id"] for b in batch],
            "z": torch.tensor([b["z"] for b in batch], device=tuned_model.device),
        }

    # Build examples as tokenized dicts (lightweight, no torch yet)
    examples = []
    for r in rows:
        x = r.get("x_eval", r.get("x_train", r.get("x_raw")))
        y = r["y"]
        tok = build_inputs_and_labels(tokenizer, x, y, ctx_len=cfg.ctx_len)
        examples.append({"id": r["id"], "z": int(r.get("z", -1)), **tok})

    # Mini-batch loop
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i:i+batch_size])
        out = tuned_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        stats = _masked_token_stats(out.logits, batch["labels"], cfg.topk, cfg.nll_quantiles)

        # Uplift: compute stats on base model, then base - tuned for select groups
        uplift_vals = {}
        if cfg.use_uplift:
            if uplift_base_model is None:
                # If there is no LoRA dir, tuned==base; uplift is zero.
                for key in ["nll_mean"] + [f"top{k}_rate" for k in cfg.topk] + ["gap_mean"]:
                    uplift_vals[key] = torch.zeros_like(stats[key])
            else:
                bout = uplift_base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                bstats = _masked_token_stats(bout.logits, batch["labels"], cfg.topk, cfg.nll_quantiles)
                for key in ["nll_mean"] + [f"top{k}_rate" for k in cfg.topk] + ["gap_mean"]:
                    uplift_vals[key] = bstats[key] - stats[key]

        # Stack features in order
        feats = []
        for name in ["nll_mean"] + [f"nll_q{int(q*100):02d}" for q in cfg.nll_quantiles] +                         [f"top{k}_rate" for k in cfg.topk] + ["gap_mean", "logp_sum"]:
            feats.append(stats[name].float().detach().cpu().numpy())
        if cfg.use_uplift:
            for key in ["nll_mean"] + [f"top{k}_rate" for k in cfg.topk] + ["gap_mean"]:
                feats.append(uplift_vals[key].float().detach().cpu().numpy())
        Xb = np.stack(feats, axis=1).astype(np.float32)  # [B, d]

        feats_all.append(Xb)
        labels_all.extend(batch["z"].detach().cpu().tolist())
        ids_all.extend(batch["id"])

    X = np.concatenate(feats_all, axis=0)
    y = np.array(labels_all, dtype=np.int64)
    return X, y, ids_all, f_names

def load_rows(jsonl_path: str) -> List[Dict[str, Any]]:
    return read_jsonl(jsonl_path)
