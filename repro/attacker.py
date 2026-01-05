from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import Standardizer, compute_standardizer

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, emb_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)

def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SupCon loss with cosine similarity.

    z: [B, D] projected embeddings (not yet normalized)
    y: [B] int64 labels in {0,1}
    """
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / temperature  # [B,B]
    # remove self-contrast
    logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
    sim = sim * logits_mask + (-1e9) * (1 - logits_mask)

    y = y.view(-1, 1)
    pos_mask = (y == y.t()).float() * logits_mask  # [B,B]

    # For each anchor, average over positives
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # If an anchor has no positives in batch, ignore (rare with big batches)
    pos_cnt = pos_mask.sum(dim=1)
    loss_i = -(pos_mask * log_prob).sum(dim=1) / (pos_cnt + 1e-12)
    loss = torch.where(pos_cnt > 0, loss_i, torch.zeros_like(loss_i)).mean()
    return loss

class LinearHead(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.fc(u).squeeze(-1)

@dataclass
class AttackerArtifacts:
    standardizer: Standardizer
    feature_names: Optional[list] = None

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "standardize.json"), "w", encoding="utf-8") as f:
            json.dump(self.standardizer.to_json(), f, ensure_ascii=False, indent=2)
        if self.feature_names is not None:
            with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
                json.dump(self.feature_names, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(attacker_dir: str) -> "AttackerArtifacts":
        with open(os.path.join(attacker_dir, "standardize.json"), "r", encoding="utf-8") as f:
            std = Standardizer.from_json(json.load(f))
        feature_names = None
        fn_path = os.path.join(attacker_dir, "feature_names.json")
        if os.path.exists(fn_path):
            with open(fn_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
        return AttackerArtifacts(standardizer=std, feature_names=feature_names)

def train_attacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: str,
    hidden: int = 256,
    emb_dim: int = 128,
    proj_dim: int = 64,
    batch_size: int = 256,
    epochs: int = 30,
    lr: float = 1e-3,
    temperature: float = 0.1,
    device: str = "cuda",
    feature_names: Optional[list] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Standardize using shadow train split
    std = compute_standardizer(X_train)
    Xtr = std.transform(X_train)
    Xva = std.transform(X_val)

    # Torch tensors
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch_size, shuffle=False)

    encoder = MLPEncoder(in_dim=Xtr.shape[1], hidden=hidden, emb_dim=emb_dim).to(device)
    proj = ProjectionHead(emb_dim=emb_dim, proj_dim=proj_dim).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=lr)

    # Stage A: supervised contrastive training
    encoder.train(); proj.train()
    for ep in range(1, epochs + 1):
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            u = encoder(xb)
            z = proj(u)
            loss = supervised_contrastive_loss(z, yb, temperature=temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep % 5 == 0 or ep == 1:
            print(f"[supcon] epoch={ep} loss={sum(losses)/max(1,len(losses)):.4f}")

    # Freeze encoder; train linear head
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    head = LinearHead(emb_dim=emb_dim).to(device)
    opt_h = torch.optim.AdamW(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, max(10, epochs//2) + 1):
        head.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).float()
            with torch.no_grad():
                u = encoder(xb)
            logits = head(u)
            loss = bce(logits, yb)
            opt_h.zero_grad(set_to_none=True)
            loss.backward()
            opt_h.step()
            losses.append(loss.item())
        if ep % 5 == 0 or ep == 1:
            print(f"[head] epoch={ep} loss={sum(losses)/max(1,len(losses)):.4f}")

    # Save artifacts
    torch.save(encoder.state_dict(), os.path.join(out_dir, "encoder.pt"))
    torch.save(head.state_dict(), os.path.join(out_dir, "linear_head.pt"))

    AttackerArtifacts(standardizer=std, feature_names=feature_names).save(out_dir)

def load_models(attacker_dir: str, in_dim: int, hidden: int = 256, emb_dim: int = 128, device: str = "cuda"):
    encoder = MLPEncoder(in_dim=in_dim, hidden=hidden, emb_dim=emb_dim).to(device)
    head = LinearHead(emb_dim=emb_dim).to(device)
    encoder.load_state_dict(torch.load(os.path.join(attacker_dir, "encoder.pt"), map_location=device))
    head.load_state_dict(torch.load(os.path.join(attacker_dir, "linear_head.pt"), map_location=device))
    encoder.eval(); head.eval()
    artifacts = AttackerArtifacts.load(attacker_dir)
    return encoder, head, artifacts
