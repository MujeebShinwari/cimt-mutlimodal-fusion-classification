"""Training of the single-modality base learners (the fusion arm is handled in fusion.py)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import dataset
import model as model_module
import utils


def build_criterion():
    if config.USE_FOCAL_LOSS:
        return utils.FocalLoss(gamma=config.FOCAL_LOSS_GAMMA)
    return nn.CrossEntropyLoss(label_smoothing=config.LOSS_LABEL_SMOOTHING)


def run_one_epoch(model, loader, criterion, optimizer, device, mode: str, train: bool):
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    torch.set_grad_enabled(train)

    for batch in loader:
        images = batch["image"].to(device) if mode == "image_only" else None
        demographics = batch["demographics"].to(device) if mode == "demographic_only" else None
        biomarkers = batch["biomarkers"].to(device) if mode == "biomarker_only" else None
        labels = batch["label"].to(device)

        if train:
            optimizer.zero_grad()

        logits = model(image=images, demographics=demographics, biomarkers=biomarkers)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM
            )
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    metrics = utils.compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def _train_single_arm(model_mode: str, epochs: int = None, patience: int = None,
                      logger=None, log_as_internal_pretrain: bool = False) -> Path:
    """Train one base learner and save the checkpoint with the best validation AUROC."""
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    epochs = epochs if epochs is not None else config.EPOCHS
    patience = patience if patience is not None else config.EARLY_STOPPING_PATIENCE

    if logger is None:
        logger = utils.get_logger("train", config.LOG_DIR / f"train_{model_mode}.log")

    tag = "[internal pretrain] " if log_as_internal_pretrain else ""
    logger.info(f"{tag}Starting training | model_mode={model_mode} device={device}")

    datasets, _ = dataset.prepare_datasets(mode=model_mode, fit_scaler=True)
    logger.info(f"Split sizes -> train: {len(datasets['train'])}, "
                f"val: {len(datasets['val'])}, test: {len(datasets['test'])}")

    train_loader = DataLoader(datasets["train"], batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(datasets["val"], batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    model = model_module.build_model(mode=model_mode, load_pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params / 1e6:.2f}M")

    criterion = build_criterion()

    if model_mode == "image_only":
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("image_backbone")],
             "lr": config.LR_HEAD},
            {"params": [p for n, p in model.named_parameters() if n.startswith("image_backbone")],
             "lr": config.LR_BACKBONE},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.LR_HEAD}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    early_stopping = utils.EarlyStopping(patience=patience)
    best_ckpt_path = config.CHECKPOINT_DIR / f"best_{model_mode}.pt"

    for epoch in range(epochs):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, model_mode, train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, model_mode, train=False)

        logger.info(
            f"{tag}Epoch {epoch + 1}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} auroc={train_metrics['auroc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} auroc={val_metrics['auroc']:.4f} "
            f"acc={val_metrics['accuracy']:.4f}"
        )

        early_stopping.step(val_metrics["auroc"])
        if early_stopping.improved:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_mode": model_mode,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, best_ckpt_path)
            logger.info(f"  -> new best model saved (val AUROC={val_metrics['auroc']:.4f})")

        if early_stopping.should_stop:
            logger.info(f"Early stopping at epoch {epoch + 1} (no val AUROC improvement for {patience} epochs)")
            break

    logger.info(f"{tag}Training complete. Best val AUROC = {early_stopping.best_score:.4f}. "
                f"Checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def run_training(model_mode: str = None) -> Path:
    model_mode = model_mode or config.MODEL_MODE
    if model_mode == "fusion":
        import fusion
        return fusion.run_fusion_training()
    return _train_single_arm(model_mode)


if __name__ == "__main__":
    run_training()
