"""Late-fusion stacking.

The image, demographic and biomarker models are trained independently. Each
outputs a probability of CIMT thickening, and a logistic regression over the
three probabilities (three weights and an intercept) gives the fused
prediction. The meta-learner is fitted on the validation split and evaluated
on the test split.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
import dataset
import model as model_module
import utils

BASE_MODES = ("image_only", "demographic_only", "biomarker_only")
FEATURE_COLUMNS = ["p_image", "p_demo", "p_bio"]
META_LEARNER_PATH = config.CHECKPOINT_DIR / "fusion_meta_learner.pkl"
META_LEARNER_INFO_PATH = config.OUTPUT_DIR / "fusion_meta_learner_info.json"
VAL_PREDICTIONS_PATH = config.OUTPUT_DIR / "val_predictions_fusion.csv"


def _short(mode: str) -> str:
    return {"image_only": "image", "demographic_only": "demo", "biomarker_only": "bio"}[mode]


@torch.no_grad()
def _extract_probs_for_mode(mode: str, split_name: str, device: torch.device) -> pd.DataFrame:
    """Predicted probability of one base learner for every image of a split."""
    ckpt_path = config.CHECKPOINT_DIR / f"best_{mode}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run 'python main.py --stage all --model_mode {mode}' first "
            f"(demographic_only and biomarker_only are trained automatically by the fusion arm)."
        )

    datasets, _ = dataset.prepare_datasets(mode=mode, fit_scaler=True)
    loader = DataLoader(datasets[split_name], batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    net = model_module.build_model(mode=mode, load_pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device).eval()

    keys, labels, probs, ages, genders, true_ages = [], [], [], [], [], []
    for batch in loader:
        images = batch["image"].to(device) if mode == "image_only" else None
        demographics = batch["demographics"].to(device) if mode == "demographic_only" else None
        biomarkers = batch["biomarkers"].to(device) if mode == "biomarker_only" else None

        logits = net(image=images, demographics=demographics, biomarkers=biomarkers)
        batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        keys.extend(f"{pid}_{eye}" for pid, eye in zip(batch["patient_id"], batch["eye"]))
        labels.extend(batch["label"].numpy().tolist())
        probs.extend(batch_probs.tolist())
        true_ages.extend(batch["true_age"].tolist())
        raw_demo = batch["demographics"].numpy()
        ages.extend(raw_demo[:, 0].tolist())
        genders.extend(raw_demo[:, 1].tolist())

    return pd.DataFrame({
        "key": keys, "true_label": labels, f"p_{_short(mode)}": probs,
        "true_age": true_ages, "age": ages, "gender": genders,
    }).set_index("key")


def extract_all_probabilities(split_name: str, device: torch.device) -> pd.DataFrame:
    """One row per image with p_image, p_demo, p_bio, true_label, true_age, age and gender."""
    merged = None
    for mode in BASE_MODES:
        df = _extract_probs_for_mode(mode, split_name, device)
        if merged is None:
            merged = df[[f"p_{_short(mode)}", "true_label", "true_age", "age", "gender"]]
        else:
            merged = merged.join(df[[f"p_{_short(mode)}"]], how="inner")
    return merged


def fit_meta_learner(val_df: pd.DataFrame, logger) -> LogisticRegression:
    X = val_df[FEATURE_COLUMNS].values
    y = val_df["true_label"].values

    meta = LogisticRegression(C=config.META_LEARNER_C, max_iter=1000)
    meta.fit(X, y)

    coefs = dict(zip(FEATURE_COLUMNS, meta.coef_[0].tolist()))
    logger.info(f"Meta-learner fitted on the validation split (n={len(val_df)}): "
                f"coefficients={coefs}, intercept={meta.intercept_[0]:.4f}")

    # Reference only: validation AUROC of each base learner and of a fixed
    # AUROC-weighted average of the three probabilities.
    aurocs = {c: roc_auc_score(y, val_df[c].values) for c in FEATURE_COLUMNS}
    weights = {c: max(v - 0.5, 1e-3) for c, v in aurocs.items()}
    weighted_avg = sum(weights[c] * val_df[c].values for c in FEATURE_COLUMNS) / sum(weights.values())
    weighted_avg_auroc = roc_auc_score(y, weighted_avg)
    logger.info(f"Base-learner validation AUROCs: {aurocs}; weighted-average AUROC: {weighted_avg_auroc:.4f}")

    info = {
        "logistic_regression_coefficients": coefs,
        "logistic_regression_intercept": float(meta.intercept_[0]),
        "base_learner_val_aurocs": aurocs,
        "weighted_average_weights": weights,
        "weighted_average_val_auroc_sanity_check": float(weighted_avg_auroc),
        "meta_learner_C": config.META_LEARNER_C,
        "n_val_samples": len(val_df),
    }
    with open(META_LEARNER_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    with open(META_LEARNER_PATH, "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"Saved meta-learner to {META_LEARNER_PATH} and its summary to {META_LEARNER_INFO_PATH}")
    return meta


def load_meta_learner() -> LogisticRegression:
    if not META_LEARNER_PATH.exists():
        raise FileNotFoundError(
            f"No meta-learner at {META_LEARNER_PATH}; run 'python main.py --stage train --model_mode fusion' first."
        )
    with open(META_LEARNER_PATH, "rb") as f:
        return pickle.load(f)


def _with_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "patient_id", [k.rsplit("_", 1)[0] for k in out.index])
    out.insert(1, "eye", [k.rsplit("_", 1)[1] for k in out.index])
    return out.reset_index(drop=True)


def run_fusion_training() -> Path:
    device = torch.device(config.DEVICE)
    logger = utils.get_logger("train", config.LOG_DIR / "train_fusion.log")
    logger.info(f"Starting fusion training (late-fusion stacking) | device={device}")

    if not (config.CHECKPOINT_DIR / "best_image_only.pt").exists():
        raise FileNotFoundError(
            "No best_image_only.pt checkpoint. Run 'python main.py --stage all --model_mode image_only' first."
        )

    import train as train_module  # imported here to avoid a circular import
    for internal_mode in config.INTERNAL_PRETRAIN_MODES:
        ckpt_path = config.CHECKPOINT_DIR / f"best_{internal_mode}.pt"
        if ckpt_path.exists():
            logger.info(f"[internal pretrain] found {ckpt_path.name}, skipping training")
            continue
        logger.info(f"[internal pretrain] {ckpt_path.name} not found, training it now")
        train_module._train_single_arm(internal_mode, logger=logger, log_as_internal_pretrain=True)

    logger.info("Extracting base-learner probabilities on the validation split")
    val_df = extract_all_probabilities("val", device)
    logger.info(f"Assembled {len(val_df)} validation rows")
    _with_ids(val_df).to_csv(VAL_PREDICTIONS_PATH, index=False)

    fit_meta_learner(val_df, logger)
    logger.info("Fusion training complete")
    return META_LEARNER_PATH


@torch.no_grad()
def run_fusion_testing() -> dict:
    device = torch.device(config.DEVICE)
    logger = utils.get_logger("test", config.LOG_DIR / "test_fusion.log")
    logger.info(f"Evaluating fusion (late-fusion stacking) on the test split | device={device}")

    meta = load_meta_learner()
    test_df = extract_all_probabilities("test", device)
    logger.info(f"Assembled {len(test_df)} test rows")

    fused_probs = meta.predict_proba(test_df[FEATURE_COLUMNS].values)[:, 1]
    y_test = test_df["true_label"].values

    metrics = utils.compute_metrics(y_test, fused_probs)
    logger.info("Test set results:")
    utils.print_metrics(metrics, prefix="[TEST]")
    logger.info(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    results_path = config.OUTPUT_DIR / "test_results_fusion.json"
    utils.save_json(metrics, results_path)
    logger.info(f"Saved metrics to {results_path}")

    predictions_path = config.OUTPUT_DIR / "test_predictions_fusion.csv"
    pd.DataFrame({
        "patient_id": [k.rsplit("_", 1)[0] for k in test_df.index],
        "eye": [k.rsplit("_", 1)[1] for k in test_df.index],
        "true_age": test_df["true_age"].values,
        "true_label": y_test,
        "predicted_probability": fused_probs,
        "p_image": test_df["p_image"].values,
        "p_demo": test_df["p_demo"].values,
        "p_bio": test_df["p_bio"].values,
    }).to_csv(predictions_path, index=False)
    logger.info(f"Saved per-sample predictions to {predictions_path}")
    return metrics
