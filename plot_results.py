"""Figures from the saved logs, predictions and metrics.

Usage:
    python plot_results.py --model_mode fusion --compare image_only

Reads train_<mode>.log, test_predictions_<mode>.csv and test_results_<mode>.json
from the results folder and writes PNG files to <results>/figures/.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import config

FIG_DIR = config.OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "image_only": "#5B9BD5",
    "fusion": "#ED7D31",
    "demographic_only": "#F2C511",
    "biomarker_only": "#70AD47",
}
LABELS = {
    "image_only": "Image-only (RETFound)",
    "fusion": "Fusion (stacked: image + demographics + biomarkers)",
    "demographic_only": "Demographic-only (age, gender)",
    "biomarker_only": "Biomarker-only (MLP)",
}

EPOCH_LINE = re.compile(
    r"Epoch (\d+)/(\d+) \| train loss=([\d.]+) auroc=([\d.]+) \| "
    r"val loss=([\d.]+) auroc=([\d.]+) acc=([\d.]+)"
)


def parse_training_log(model_mode: str) -> pd.DataFrame:
    log_path = config.LOG_DIR / f"train_{model_mode}.log"
    rows = []
    with open(log_path) as f:
        for line in f:
            m = EPOCH_LINE.search(line)
            if m:
                epoch, total, tr_loss, tr_auroc, val_loss, val_auroc, val_acc = m.groups()
                rows.append({
                    "epoch": int(epoch), "train_loss": float(tr_loss), "train_auroc": float(tr_auroc),
                    "val_loss": float(val_loss), "val_auroc": float(val_auroc), "val_acc": float(val_acc),
                })
    return pd.DataFrame(rows)


def plot_training_curves(model_mode: str) -> list:
    df = parse_training_log(model_mode)
    if df.empty:
        print(f"[skip] no epoch lines found in train_{model_mode}.log")
        return []
    saved = []

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss", color="#5B9BD5")
    ax.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss", color="#ED7D31")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Training and validation loss ({LABELS.get(model_mode, model_mode)})")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / f"loss_curve_{model_mode}.png"; fig.savefig(p, dpi=200); plt.close(fig); saved.append(p)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(df["epoch"], df["train_auroc"], marker="o", label="Train AUROC", color="#5B9BD5")
    ax.plot(df["epoch"], df["val_auroc"], marker="o", label="Validation AUROC", color="#ED7D31")
    best_epoch = df.loc[df["val_auroc"].idxmax(), "epoch"]; best_auroc = df["val_auroc"].max()
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    ax.annotate(f"best val AUROC={best_auroc:.3f}\n(epoch {int(best_epoch)})",
                xy=(best_epoch, best_auroc), xytext=(best_epoch + 0.5, best_auroc - 0.08),
                fontsize=9, color="#444444")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUROC")
    ax.set_title(f"Training and validation AUROC ({LABELS.get(model_mode, model_mode)})")
    ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / f"auroc_curve_{model_mode}.png"; fig.savefig(p, dpi=200); plt.close(fig); saved.append(p)
    return saved


def load_predictions(model_mode: str) -> pd.DataFrame:
    return pd.read_csv(config.OUTPUT_DIR / f"test_predictions_{model_mode}.csv")


def plot_roc_curves(model_modes: list) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for mm in model_modes:
        df = load_predictions(mm)
        fpr, tpr, _ = roc_curve(df["true_label"], df["predicted_probability"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{LABELS.get(mm, mm)} (AUROC={roc_auc:.3f})", color=COLORS.get(mm), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)"); ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve: CIMT classification (test set)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    p = FIG_DIR / "roc_curve.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return p


def plot_pr_curves(model_modes: list) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for mm in model_modes:
        df = load_predictions(mm)
        precision, recall, _ = precision_recall_curve(df["true_label"], df["predicted_probability"])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{LABELS.get(mm, mm)} (AUC={pr_auc:.3f})", color=COLORS.get(mm), linewidth=2)
    prevalence = load_predictions(model_modes[0])["true_label"].mean()
    ax.axhline(prevalence, linestyle="--", color="grey", linewidth=1, label=f"Chance (prevalence={prevalence:.2f})")
    ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve: CIMT classification (test set)")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    p = FIG_DIR / "pr_curve.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return p


def plot_confusion_matrix(model_mode: str) -> Path:
    with open(config.OUTPUT_DIR / f"test_results_{model_mode}.json") as f:
        results = json.load(f)
    cm = np.array(results["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color=color, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Thickened"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Thickened"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {LABELS.get(model_mode, model_mode)}\n(test set)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = FIG_DIR / f"confusion_matrix_{model_mode}.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return p


def plot_probability_distribution(model_mode: str) -> Path:
    df = load_predictions(model_mode)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bins = np.linspace(0, 1, 21)
    ax.hist(df.loc[df["true_label"] == 0, "predicted_probability"], bins=bins, alpha=0.65, label="True: Normal", color="#70AD47")
    ax.hist(df.loc[df["true_label"] == 1, "predicted_probability"], bins=bins, alpha=0.65, label="True: Thickened", color="#C0392B")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted probability of thickened CIMT"); ax.set_ylabel("Number of test images")
    ax.set_title(f"Predicted-probability distribution\n({LABELS.get(model_mode, model_mode)})")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = FIG_DIR / f"probability_distribution_{model_mode}.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return p


def plot_calibration(model_mode: str, n_bins: int = 8) -> Path:
    df = load_predictions(model_mode)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(df["predicted_probability"], bin_edges) - 1, 0, n_bins - 1)
    mean_pred, mean_true, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(df.loc[mask, "predicted_probability"].mean())
        mean_true.append(df.loc[mask, "true_label"].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    sizes = np.array(counts) * 8
    ax.scatter(mean_pred, mean_true, s=sizes, color="#ED7D31", edgecolor="white", zorder=3,
               label="Observed (marker size = # images in bin)")
    ax.plot(mean_pred, mean_true, color="#ED7D31", linewidth=1, alpha=0.6)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction thickened")
    ax.set_title(f"Calibration (reliability) curve\n({LABELS.get(model_mode, model_mode)})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = FIG_DIR / f"calibration_{model_mode}.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return p


def plot_metric_bar_chart(model_modes: list) -> Path:
    metrics = ["accuracy", "auroc", "f1", "sensitivity", "specificity"]
    data = {}
    for mm in model_modes:
        with open(config.OUTPUT_DIR / f"test_results_{mm}.json") as f:
            data[mm] = json.load(f)

    x = np.arange(len(metrics))
    width = 0.8 / len(model_modes)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, mm in enumerate(model_modes):
        values = [data[mm][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=LABELS.get(mm, mm), color=COLORS.get(mm))
    ax.set_xticks(x + width * (len(model_modes) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Test-set performance by model configuration")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "metric_comparison_bar.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return p


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from saved outputs.")
    parser.add_argument("--model_mode", type=str, default="fusion")
    parser.add_argument("--compare", type=str, nargs="*", default=[])
    args = parser.parse_args()

    all_modes = [args.model_mode] + [m for m in args.compare if m != args.model_mode]

    print("Training curves...")
    for p in plot_training_curves(args.model_mode):
        print(f"  saved {p}")

    print("Confusion matrix...")
    print(f"  saved {plot_confusion_matrix(args.model_mode)}")

    print("Probability distribution...")
    print(f"  saved {plot_probability_distribution(args.model_mode)}")

    print("Calibration curve...")
    print(f"  saved {plot_calibration(args.model_mode)}")

    available = [m for m in all_modes if (config.OUTPUT_DIR / f"test_predictions_{m}.csv").exists()]
    if available:
        print(f"ROC curve (modes: {available})...")
        print(f"  saved {plot_roc_curves(available)}")
        print(f"PR curve (modes: {available})...")
        print(f"  saved {plot_pr_curves(available)}")
        print(f"Metric comparison bar chart (modes: {available})...")
        print(f"  saved {plot_metric_bar_chart(available)}")
    else:
        print("No test_predictions_<mode>.csv files found for ROC/PR/bar comparison.")

    print(f"\nAll figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
