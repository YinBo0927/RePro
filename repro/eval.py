from __future__ import annotations
import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # TPR@1%FPR
    target = 0.01
    idx = np.where(fpr <= target)[0]
    tpr_at = float(tpr[idx].max()) if len(idx) else float(tpr[0])
    return {"auc": auc, "tpr_at_1pct_fpr": tpr_at}

def choose_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # smallest threshold with fpr <= target, prefer highest tpr
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return float(thr[-1])
    best = idx[np.argmax(tpr[idx])]
    return float(thr[best])

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_png: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xscale("log")
    plt.grid(True, which="both")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
