from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset

from .utils import read_jsonl, write_jsonl, set_seed

def prepare_raw(dataset: str) -> List[Dict[str, Any]]:
    """Load benchmark and return rows with {id, x_raw, y}."""
    rows: List[Dict[str, Any]] = []
    if dataset.lower() == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        # Use the canonical training split as the instance pool; you'll split later.
        for i, ex in enumerate(ds["train"]):
            rows.append({
                "id": f"gsm8k-{i}",
                "x_raw": ex["question"].strip(),
                "y": ex["answer"].strip(),
            })
    elif dataset.lower() == "humaneval":
        ds = load_dataset("openai_humaneval")
        for i, ex in enumerate(ds["test"]):
            # HumanEval prompt typically includes signature + docstring stub.
            x_raw = ex["prompt"].rstrip()
            # Canonical solution is the supervised reference output for teacher forcing.
            y = ex["canonical_solution"].rstrip() + "\n"
            rows.append({
                "id": f"humaneval-{i}",
                "x_raw": x_raw,
                "y": y,
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return rows

def merge_raw_ref(raw_path: str, ref_path: str) -> List[Dict[str, Any]]:
    raw = {r["id"]: r for r in read_jsonl(raw_path)}
    ref = {r["id"]: r for r in read_jsonl(ref_path)}
    out = []
    for k, r in raw.items():
        if k not in ref:
            raise KeyError(f"Missing refined prompt for id={k}")
        out.append({
            "id": k,
            "x_raw": r["x_raw"],
            "x_ref": ref[k]["x_ref"],
            "y": r["y"],
        })
    return out

def split_instance_disjoint(rows: List[Dict[str, Any]], seed: int, shadow_frac: float = 0.5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into shadow pool and victim pool with no overlap at instance level."""
    set_seed(seed)
    idx = np.arange(len(rows))
    np.random.shuffle(idx)
    cut = int(len(idx) * shadow_frac)
    shadow = [rows[i] for i in idx[:cut]]
    victim = [rows[i] for i in idx[cut:]]
    return shadow, victim

def sample_mixture(rows: List[Dict[str, Any]], rho: float, seed: int) -> List[Dict[str, Any]]:
    """Sample z ~ Bernoulli(rho) per instance; fix once."""
    set_seed(seed)
    out = []
    for r in rows:
        z = int(np.random.rand() < rho)
        x_train = r["x_ref"] if z == 1 else r["x_raw"]
        out.append({
            **r,
            "z": z,
            "x_train": x_train,
        })
    return out

def train_val_test_split(rows: List[Dict[str, Any]], seed: int, val_frac: float = 0.1, test_frac: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    set_seed(seed)
    idx = np.arange(len(rows))
    np.random.shuffle(idx)
    n = len(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = [rows[i] for i in idx[:n_test]]
    val = [rows[i] for i in idx[n_test:n_test + n_val]]
    train = [rows[i] for i in idx[n_test + n_val:]]
    return train, val, test

def build_mixtures(raw_path: str, ref_path: str, out_dir: str, rho: float, seed: int) -> None:
    merged = merge_raw_ref(raw_path, ref_path)
    shadow_pool, victim_pool = split_instance_disjoint(merged, seed=seed, shadow_frac=0.5)

    shadow_mix = sample_mixture(shadow_pool, rho=rho, seed=seed + 1)
    victim_mix = sample_mixture(victim_pool, rho=rho, seed=seed + 2)

    shadow_train, shadow_val, shadow_test = train_val_test_split(shadow_mix, seed=seed + 3)
    victim_train, victim_val, victim_test = train_val_test_split(victim_mix, seed=seed + 4)

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "shadow_train.jsonl"), shadow_train)
    write_jsonl(os.path.join(out_dir, "shadow_val.jsonl"), shadow_val)
    write_jsonl(os.path.join(out_dir, "shadow_test.jsonl"), shadow_test)

    write_jsonl(os.path.join(out_dir, "victim_train.jsonl"), victim_train)
    write_jsonl(os.path.join(out_dir, "victim_val.jsonl"), victim_val)
    write_jsonl(os.path.join(out_dir, "victim_test.jsonl"), victim_test)
