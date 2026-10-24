"""Group fairness metrics for the approve, review and decline decisions.

For each group (for example women and men, or an age band) I measure:
  approval rate             share of applicants approved automatically
  default rate if approved  share of approved applicants who went on to default
  TPR                       share of applicants who defaulted that the rule caught (sent to review or declined)
  approval rate if repaid   share of applicants who repaid that the rule approved (the equal opportunity view)
  mean PD and default rate  so calibration within the group can be checked

Review counts as "not approved" here, since a reviewed applicant doesn't get an automatic yes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED

METRICS = ("approval_rate", "default_rate_approved", "tpr", "approval_rate_repaid", "mean_pd", "default_rate")
METRIC_LABELS = {
    "approval_rate": "Approval rate",
    "default_rate_approved": "Default rate among approved",
    "tpr": "TPR (defaulters not auto-approved)",
    "approval_rate_repaid": "Approval rate among those who repaid",
    "mean_pd": "Mean predicted PD",
    "default_rate": "Actual default rate",
}

AGE_EDGES = [0, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["Under 25", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "65 and over"]


def age_band(age) -> pd.Categorical:
    return pd.cut(pd.Series(age, dtype=float), AGE_EDGES, right=False, labels=AGE_LABELS)


def wilson(k, n, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion k / n; behaves better than the normal one near 0 and 1."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def group_rates(codes: np.ndarray, n_groups: int, y: np.ndarray, p: np.ndarray, approved: np.ndarray) -> dict:
    """Every metric for every group at once, using bincount so the bootstrap stays fast.

    codes holds a group number from 0 to n_groups - 1 for each applicant.
    """
    def count(weights=None):
        return np.bincount(codes, weights=weights, minlength=n_groups)

    y = y.astype(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "approval_rate": count(approved) / count(),
            "default_rate_approved": count(approved & y) / count(approved),
            "tpr": count(~approved & y) / count(y),
            "approval_rate_repaid": count(approved & ~y) / count(~y),
            "mean_pd": count(p) / count(),
            "default_rate": count(y) / count(),
        }


def _codes(groups: pd.Series) -> tuple[np.ndarray, list[str]]:
    # groups with no applicants (unused categories) are dropped, and missing values are left out with code -1;
    # an ordered category like the age bands keeps its order
    if isinstance(groups.dtype, pd.CategoricalDtype):
        cat = pd.Categorical(groups).remove_unused_categories()
    else:
        cat = pd.Categorical(groups.astype("object"))
    return cat.codes.astype(int), [str(c) for c in cat.categories]


def group_table(groups: pd.Series, y, p, decision) -> pd.DataFrame:
    """One row per group with counts, every metric and Wilson intervals for the decision rates."""
    codes, labels = _codes(groups)
    keep = codes >= 0
    y = np.asarray(y)[keep].astype(bool)
    p = np.asarray(p, dtype=float)[keep]
    approved = np.asarray(decision)[keep] == "approve"
    codes = codes[keep]
    rates = group_rates(codes, len(labels), y, p, approved)
    rows = []
    for g, label in enumerate(labels):
        m = codes == g
        row = {"group": label, "n": int(m.sum()), "share": float(m.mean())}
        row.update({k: float(v[g]) for k, v in rates.items()})
        # Wilson intervals, each on its own denominator
        for metric, num, den in [("approval_rate", approved[m], m[m]),
                                 ("default_rate_approved", y[m & approved], approved[m]),
                                 ("tpr", ~approved[m & y], y[m]),
                                 ("approval_rate_repaid", approved[m & ~y], ~y[m])]:
            row[f"{metric}_low"], row[f"{metric}_high"] = wilson(int(num.sum()), int(den.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def gap_summary(groups: pd.Series, y, p_main, p_alt, decide_fn, n_boot: int = 1000, seed: int = SEED,
                min_n: int = 30) -> pd.DataFrame:
    """Largest gap between groups (highest minus lowest) for each metric, for two models.

    Returns the gap under the main model, under the alternative and the change, each with a paired
    bootstrap 95% interval: both models are scored on the same resampled applicants, so the interval for
    the change is much tighter than the two separate intervals suggest. Groups smaller than min_n are left out.
    """
    codes, labels = _codes(groups)
    sizes = np.bincount(codes[codes >= 0], minlength=len(labels))
    keep_groups = np.flatnonzero(sizes >= min_n)
    keep = np.isin(codes, keep_groups)
    remap = np.full(len(labels), -1)
    remap[keep_groups] = np.arange(len(keep_groups))
    codes = remap[codes[keep]]
    n_groups = len(keep_groups)
    y = np.asarray(y)[keep].astype(bool)
    p_main = np.asarray(p_main, dtype=float)[keep]
    p_alt = np.asarray(p_alt, dtype=float)[keep]
    appr_main = np.asarray(decide_fn(p_main)) == "approve"
    appr_alt = np.asarray(decide_fn(p_alt)) == "approve"

    def gaps(idx):
        out = {}
        for name, p, appr in [("main", p_main, appr_main), ("alt", p_alt, appr_alt)]:
            rates = group_rates(codes[idx], n_groups, y[idx], p[idx], appr[idx])
            out[name] = {k: np.nanmax(v) - np.nanmin(v) for k, v in rates.items()}
        return out

    point = gaps(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = {m: {"main": [], "alt": [], "change": []} for m in METRICS}
    for _ in range(n_boot):
        g = gaps(rng.integers(0, len(y), len(y)))
        for m in METRICS:
            draws[m]["main"].append(g["main"][m])
            draws[m]["alt"].append(g["alt"][m])
            draws[m]["change"].append(g["alt"][m] - g["main"][m])
    rows = []
    for m in METRICS:
        row = {"metric": m, "main": point["main"][m], "alt": point["alt"][m],
               "change": point["alt"][m] - point["main"][m]}
        for k in ("main", "alt", "change"):
            row[f"{k}_low"], row[f"{k}_high"] = np.quantile(draws[m][k], [0.025, 0.975])
        rows.append(row)
    return pd.DataFrame(rows)
