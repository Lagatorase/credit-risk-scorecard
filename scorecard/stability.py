"""Bootstrap confidence intervals and the population stability index (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED
from .metrics import all_metrics


def bootstrap_ci(y, p, n_boot: int = 1000, alpha: float = 0.05, seed: int = SEED) -> pd.DataFrame:
    """Point estimate and percentile 95% interval for every metric, resampling applicants with replacement."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    point = all_metrics(y, p)
    draws = {name: [] for name in point}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for name, value in all_metrics(y[idx], p[idx]).items():
            draws[name].append(value)
    rows = []
    for name, value in point.items():
        lo, hi = np.quantile(draws[name], [alpha / 2, 1 - alpha / 2])
        rows.append({"metric": name, "value": value, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index between a reference sample (expected) and a new one (actual).

    Bins are deciles of the reference sample, and missing values get their own bin. For categorical
    values each category is a bin. Common reading: under 0.1 stable, 0.1 to 0.25 watch, over 0.25 shift.
    """
    e = pd.Series(expected)
    a = pd.Series(actual)
    if pd.api.types.is_numeric_dtype(e):
        e = e.astype(float)
        a = a.astype(float)
        cuts = np.unique(np.nanquantile(e, np.linspace(0, 1, n_bins + 1)[1:-1]))
        e_bins = np.where(e.isna(), -1, np.searchsorted(cuts, e.fillna(0), side="right"))
        a_bins = np.where(a.isna(), -1, np.searchsorted(cuts, a.fillna(0), side="right"))
    else:
        e_bins = e.astype("object").fillna("Missing").to_numpy()
        a_bins = a.astype("object").fillna("Missing").to_numpy()
    labels = np.union1d(np.unique(e_bins), np.unique(a_bins))
    e_share = pd.Series(e_bins).value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    a_share = pd.Series(a_bins).value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    # a small floor stops empty bins from giving infinite values
    e_share = np.maximum(e_share, eps)
    a_share = np.maximum(a_share, eps)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_reading(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value <= 0.25:
        return "watch"
    return "shift"
