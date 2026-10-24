"""Ranking and calibration metrics used in every experiment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def gini(y, p) -> float:
    # Gini is just a rescaled AUC that credit risk teams prefer: 0 is random, 1 is perfect
    return float(2 * roc_auc_score(y, p) - 1)


def ks(y, p) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-frequency bins.

    With an 8% default rate, equal-width bins would put most applicants in the first bin,
    so I use deciles of the predicted probability instead. ECE is the row-weighted average gap
    between the mean prediction and the actual default rate in each bin.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    gaps = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            gaps += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(gaps / len(y))


def all_metrics(y, p) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "gini": gini(y, p),
        "ks": ks(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def fold_summary(y, fold_preds) -> dict[str, float]:
    """Mean and standard deviation of every metric across the CV folds."""
    per_fold = [all_metrics(y[idx], p) for idx, p in fold_preds]
    out = {}
    for name in per_fold[0]:
        vals = np.array([m[name] for m in per_fold])
        out[f"{name}_mean"] = float(vals.mean())
        out[f"{name}_std"] = float(vals.std(ddof=1))
    return out
