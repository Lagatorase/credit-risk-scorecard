"""Reason codes: the top factors that pushed an applicant's risk up, in plain English.

The factors come from SHAP values of the final model. The text for each feature comes from
config/reason_codes.yaml, which must cover every model feature (a test checks this).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ROOT

REASON_CODES = ROOT / "config" / "reason_codes.yaml"


def load_reason_map(path: Path = REASON_CODES) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def customer_allowed(entry: dict) -> bool:
    # every reason can go to the loan officer; customer: false keeps it out of the customer's version
    return entry.get("customer", True)


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:]


def reason_text(entry: dict, value, median: float | None) -> str:
    """The plain-English text for one feature value that increased risk."""
    missing = pd.isna(value)
    if "values" in entry or "text" in entry:
        # categorical: one text per category, or a template with the category filled in
        if missing:
            return entry.get("missing", _sentence(entry.get("text", "{value}").format(value="not stated")))
        if "values" in entry:
            return entry["values"].get(str(value), f"{value}")
        return entry["text"].format(value=value)
    if missing:
        default = _sentence(f"{entry['name']} not available") if "name" in entry else "Information not available"
        return entry.get("missing", default)
    if "flag_yes" in entry:
        return entry["flag_yes"] if float(value) >= 0.5 else entry["flag_no"]
    # numeric: compare with the development median to say whether the value is high or low
    side = "high" if median is not None and float(value) > median else "low"
    if side in entry:
        return entry[side]
    return f"{side.capitalize()} {entry['name']}"


def _plain_value(value):
    # a JSON-friendly version of the raw value: None for missing, float for numbers, text for categories
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return str(value)


def top_reasons(shap_row: np.ndarray, x_row: pd.Series, features: list[str], medians: dict[str, float],
                mapping: dict[str, dict], k: int = 4, audience: str = "officer") -> list[dict]:
    """The k features with the largest positive SHAP values (pushing the PD up), as reason codes.

    For the customer audience, features marked customer: false (age) are skipped and the next ones are used.
    """
    order = np.argsort(-np.asarray(shap_row))
    reasons = []
    for i in order:
        impact = float(shap_row[i])
        if impact <= 0 or len(reasons) == k:
            break
        feature = features[i]
        entry = mapping[feature]
        if audience == "customer" and not customer_allowed(entry):
            continue
        value = x_row[feature]
        reasons.append({
            "feature": feature,
            "value": _plain_value(value),
            "impact": impact,
            "direction": "increases risk",
            "text": reason_text(entry, value, medians.get(feature)),
        })
    return reasons
