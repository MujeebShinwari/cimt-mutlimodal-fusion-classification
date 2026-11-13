"""Modality ablation for the late-fusion meta-learner.

A logistic regression is fitted on the validation probabilities of each subset
of modalities, evaluated on the test split, and compared with the full
three-modality fusion (DeLong test, patient-level bootstrap CI).

Usage:
    python ablation_modality_combinations.py VAL_CSV TEST_CSV [--n_boot 5000]

VAL_CSV  = results/val_predictions_fusion.csv
TEST_CSV = results/test_predictions_fusion.csv
Both need the columns patient_id, true_label, p_image, p_demo, p_bio.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

COMBINATIONS = {
    "Image only": ["p_image"],
    "Demographics only": ["p_demo"],
    "Biomarkers only": ["p_bio"],
    "Image + demographics": ["p_image", "p_demo"],
    "Image + biomarkers": ["p_image", "p_bio"],
    "Demographics + biomarkers": ["p_demo", "p_bio"],
    "All three (fusion)": ["p_image", "p_demo", "p_bio"],
}


def delong_test(y, a, b):
    """Two-sided DeLong test for the difference between two correlated AUROCs."""
    pos = y == 1

    def components(s):
        p, n = s[pos], s[~pos]
        psi = (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])
        return psi.mean(), psi.mean(1), psi.mean(0)

    auc_a, a10, a01 = components(a)
    auc_b, b10, b01 = components(b)
    cov = np.cov(np.vstack([a10, b10])) / pos.sum() + np.cov(np.vstack([a01, b01])) / (~pos).sum()
    z = (auc_a - auc_b) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return z, 2 * stats.norm.sf(abs(z))


def patient_bootstrap_indices(test, n_boot, seed):
    """Resample patients (both eyes together), stratified by class."""
    rows = {p: np.where(test.patient_id.values == p)[0] for p in test.patient_id.unique()}
    pos_p = test[test.true_label == 1].patient_id.unique()
    neg_p = test[test.true_label == 0].patient_id.unique()
    rs = np.random.RandomState(seed)
    return [
        np.concatenate([rows[p] for p in np.concatenate(
            [rs.choice(pos_p, len(pos_p)), rs.choice(neg_p, len(neg_p))])])
        for _ in range(n_boot)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("val_csv")
    parser.add_argument("test_csv")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0, help="meta-learner regularization (default 1.0)")
    parser.add_argument("--out", default="ablation_modality_results.csv")
    args = parser.parse_args()

    val, test = pd.read_csv(args.val_csv), pd.read_csv(args.test_csv)
    y = test.true_label.values

    probs = {}
    for name, cols in COMBINATIONS.items():
        meta = LogisticRegression(C=args.C, max_iter=1000).fit(val[cols].values, val.true_label.values)
        probs[name] = meta.predict_proba(test[cols].values)[:, 1]

    boot_idx = patient_bootstrap_indices(test, args.n_boot, args.seed)
    full = probs["All three (fusion)"]

    rows = []
    for name, p in probs.items():
        pred = p >= 0.5
        lo, hi = np.percentile([roc_auc_score(y[i], p[i]) for i in boot_idx], [2.5, 97.5])
        p_value = np.nan if name == "All three (fusion)" else delong_test(y, full, p)[1]
        rows.append({
            "model": name,
            "AUROC": roc_auc_score(y, p), "CI_lo": lo, "CI_hi": hi,
            "sensitivity": (pred & (y == 1)).sum() / (y == 1).sum(),
            "specificity": (~pred & (y == 0)).sum() / (y == 0).sum(),
            "accuracy": (pred == y).mean(),
            "delong_p_vs_fusion": p_value,
        })

    out = pd.DataFrame(rows).round(4)
    print(out.to_string(index=False))
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
