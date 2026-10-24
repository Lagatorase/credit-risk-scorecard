"""Reason codes: the factors that pushed an applicant's risk up, worded from their actual values.

SHAP decides which features increased this applicant's PD. Each reason then states the value and
where it sits among development applicants, e.g. "Average size of previous loans: 45,000 (lowest 20%
of applicants)", so the text never claims a direction the feature doesn't consistently have.
Reasons are grouped into families and a list shows at most one reason per family.
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .config import ROOT

REASON_CODES = ROOT / "config" / "reason_codes.yaml"

# a value shared by at least this share of applicants is described by that share instead of a percentile
TIE_SHARE = 0.10


def load_reason_config(path: Path = REASON_CODES) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_reference(X: pd.DataFrame, n_quantiles: int = 1001) -> dict:
    """Where every feature's values sit among the given applicants (the development set).

    Numeric features keep 1,001 quantiles plus the share of applicants at each common value (every
    value for flags and small counts), so ties like "0 late payments" can be described exactly. Categorical features keep category shares.
    """
    ref = {}
    for col in X.columns:
        s = X[col]
        missing = float(s.isna().mean())
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna().to_numpy(dtype="float64")
            shares = pd.Series(v).value_counts(normalize=True)
            # flags and small counts keep every value, so even a rare answer gets its true share
            if len(shares) > 50:
                shares = shares[shares >= 0.01]
            ref[col] = {
                "kind": "numeric",
                "quantiles": np.quantile(v, np.linspace(0, 1, n_quantiles)) if len(v) else np.array([]),
                "mass": {float(k): float(sh) for k, sh in shares.items()},
                "missing": missing,
            }
        else:
            shares = s.astype("object").dropna().value_counts(normalize=True)
            ref[col] = {"kind": "category", "shares": {str(k): float(v) for k, v in shares.items()}, "missing": missing}
    return ref


def save_reference(ref: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, path)


def position_phrase(value: float, ref: dict) -> str:
    """Where a numeric value sits among development applicants, in tens of percent."""
    equal = ref["mass"].get(float(value), 0.0)
    if equal >= TIE_SHARE:
        # a percentile would mislead when a big block of applicants share the value (e.g. no late payments)
        return f"shared by {equal:.1%} of applicants"
    q = ref["quantiles"]
    below = min(np.searchsorted(q, value, side="left") / (len(q) - 1), 1.0)
    # mid-rank: share below, plus half of those with exactly this value
    p = below + equal / 2
    if 0.4 <= p < 0.6:
        # "lowest 50%" would be accurate but says nothing, so I say so plainly
        return "around the middle of applicants"
    if p < 0.5:
        return f"lowest {max(10, math.ceil(p * 10) * 10)}% of applicants"
    return f"highest {max(10, math.ceil((1 - p) * 10) * 10)}% of applicants"


def format_value(value, fmt: str, currency: str = "", unit: str | None = None) -> str:
    """Formats a raw value for display: money with thousands separators, percentages to one decimal place."""
    if fmt == "category":
        # Y/N flags read as Yes/No, and Home Credit's "XNA" code means not applicable (e.g. pensioners have no employer)
        return {"Y": "Yes", "N": "No", "XNA": "Not applicable"}.get(str(value), str(value))
    v = float(value)
    if fmt == "money":
        text = f"{currency}{v:,.0f}"
    elif fmt in ("percent", "share"):
        text = f"{v * 100:.1f}%"
    elif fmt == "times":
        text = f"{v:.2f} times"
    elif fmt == "years":
        text = f"{v:.1f} years"
    elif fmt == "days_ago":
        # adding 0.0 turns -0.0 into 0.0, so a value of zero doesn't print as "-0 days ago"
        days = -v + 0.0
        text = f"{days:,.0f} days ago" if days < 730 else f"{days / 365.25:.1f} years ago"
    elif fmt == "days_ahead":
        if v < 0:
            text = f"{-v:,.0f} days ago"
        else:
            text = f"in {v:,.0f} days" if v < 730 else f"in {v / 365.25:.1f} years"
    elif fmt == "count":
        text = f"{v:,.0f}"
    elif fmt == "score":
        text = f"{v:.2f}"
    elif fmt == "flag":
        text = "Yes" if v >= 0.5 else "No"
    else:
        text = f"{v:,.0f}" if v == int(v) else f"{v:,.1f}"
    return f"{text} {unit}" if unit and fmt in ("count", "number") else text


class ReasonCoder:
    """Turns SHAP values and raw feature values into reason codes for the officer or the customer."""

    def __init__(self, reference: dict, config: dict | None = None):
        self.reference = reference
        config = config or load_reason_config()
        self.features = config["features"]
        self.families = config["families"]
        self.currency = config.get("currency", "")

    def officer_only(self, feature: str) -> bool:
        return bool(self.features[feature].get("officer_only", False))

    def text(self, feature: str, value, audience: str = "officer") -> str:
        entry = self.features[feature]
        label = entry["officer"] if audience == "officer" else entry.get("customer", entry["officer"])
        ref = self.reference[feature]
        if pd.isna(value):
            return f"{label}: not available (missing for {ref['missing']:.1%} of applicants)"
        shown = format_value(value, entry["format"], self.currency, entry.get("unit"))
        if entry["format"] in ("category", "flag"):
            # for categories and yes/no flags, say how common this answer is ("shared by" so it can't be
            # misread as a percentile, e.g. "highest 2.2%")
            if ref["kind"] == "category":
                share = ref["shares"].get(str(value), 0.0)
            else:
                share = ref["mass"].get(float(value), 0.0)
            return f"{label}: {shown} (shared by {share:.1%} of applicants)"
        return f"{label}: {shown} ({position_phrase(float(value), ref)})"

    def top(self, shap_row, x_row: pd.Series, features: list[str], k: int = 4, audience: str = "officer") -> list[dict]:
        """Up to k reasons from the largest positive SHAP values, at most one per family.

        The customer list skips officer-only features and takes the next eligible one instead.
        """
        shap_row = np.asarray(shap_row)
        reasons, used_families = [], set()
        for i in np.argsort(-shap_row):
            impact = float(shap_row[i])
            if impact <= 0 or len(reasons) == k:
                break
            feature = features[i]
            family = self.features[feature]["family"]
            if family in used_families or (audience == "customer" and self.officer_only(feature)):
                continue
            used_families.add(family)
            value = x_row[feature]
            reasons.append({
                "feature": feature,
                "family": family,
                "value": None if pd.isna(value) else (str(value) if isinstance(value, str) else float(value)),
                "impact": impact,
                "direction": "increases risk",
                "text": self.text(feature, value, audience),
            })
        return reasons
