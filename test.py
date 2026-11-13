"""Evaluation of a trained model on the held-out test split."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import dataset
import model as model_module
import utils


@torch.no_grad()
def run_testing(model_mode: str = None, checkpoint_path: Path = None) -> dict:
    model_mode = model_mode or config.MODEL_MODE

    if model_mode == "fusion":
        import fusion
        return fusion.run_fusion_testing()

    device = torch.device(config.DEVICE)
    logger = utils.get_logger("test", config.LOG_DIR / f"test_{model_mode}.log")

    checkpoint_path = checkpoint_path or config.CHECKPOINT_DIR / f"best_{model_mode}.pt"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}; train this model_mode first.")

    datasets, _ = dataset.prepare_datasets(mode=model_mode, fit_scaler=False)
    test_loader = DataLoader(datasets["test"], batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    logger.info(f"Evaluating on {len(datasets['test'])} test images (model_mode={model_mode})")

    model = model_module.build_model(mode=model_mode, load_pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                f"(val AUROC {checkpoint['val_metrics']['auroc']:.4f})")

    all_labels, all_probs, all_patient_ids, all_eyes, all_true_ages = [], [], [], [], []
    for batch in test_loader:
        images = batch["image"].to(device) if model_mode == "image_only" else None
        demographics = batch["demographics"].to(device) if model_mode == "demographic_only" else None
        biomarkers = batch["biomarkers"].to(device) if model_mode == "biomarker_only" else None

        logits = model(image=images, demographics=demographics, biomarkers=biomarkers)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_labels.append(batch["label"].numpy())
        all_probs.append(probs.cpu().numpy())
        all_patient_ids.extend(batch["patient_id"])
        all_eyes.extend(batch["eye"])
        all_true_ages.extend(batch["true_age"].tolist())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = utils.compute_metrics(all_labels, all_probs)
    logger.info("Test set results:")
    utils.print_metrics(metrics, prefix="[TEST]")
    logger.info(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    results_path = config.OUTPUT_DIR / f"test_results_{model_mode}.json"
    utils.save_json(metrics, results_path)
    logger.info(f"Saved metrics to {results_path}")

    predictions_path = config.OUTPUT_DIR / f"test_predictions_{model_mode}.csv"
    pd.DataFrame({
        "patient_id": all_patient_ids,
        "eye": all_eyes,
        "true_age": all_true_ages,
        "true_label": all_labels,
        "predicted_probability": all_probs,
    }).to_csv(predictions_path, index=False)
    logger.info(f"Saved per-sample predictions to {predictions_path}")
    return metrics


if __name__ == "__main__":
    run_testing()
