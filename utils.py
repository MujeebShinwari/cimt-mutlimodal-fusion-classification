"""Seeding, logging, checkpoint loading, loss, metrics and early stopping."""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def interpolate_pos_embed(model, checkpoint_model: dict) -> None:
    """Resize the checkpoint's position embedding if the patch grid differs."""
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size == new_size:
        return

    print(f"Resizing position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def load_retfound_backbone(model, checkpoint_path: Path) -> None:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"RETFound checkpoint not found at {checkpoint_path}. "
            f"Set RETFOUND_CHECKPOINT or place the file as described in the README."
        )
    print(f"Loading RETFound weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint

    state_dict = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    interpolate_pos_embed(model, checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"RETFound weights loaded (missing keys: {len(missing)}, unexpected keys: {len(unexpected)})")


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce_loss)
        return (((1 - p_t) ** self.gamma) * ce_loss).mean()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    print(
        f"{prefix} acc={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f} "
        f"f1={metrics['f1']:.4f} sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f}"
    )


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class EarlyStopping:
    """Tracks the best validation score and signals when to stop."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False
        self.improved = False

    def step(self, score: float) -> None:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            self.improved = False
            if self.counter >= self.patience:
                self.should_stop = True
