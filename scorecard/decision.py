"""Turns a PD into approve, review or decline using expected cost.

For one applicant with loan amount L:
  expected cost of approving = PD x LGD x L          (they might default)
  expected cost of declining = (1 - PD) x margin x L (they might have repaid, and we lose the margin)
Approving is cheaper when PD < margin / (LGD + margin), the break-even PD. Both costs scale with L,
so the break-even PD is the same for every loan size. Applicants whose two costs are within
review_band x L of each other go to manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import ROOT

DECISION_CONFIG = ROOT / "config" / "decision.yaml"


@dataclass(frozen=True)
class DecisionConfig:
    lgd: float
    margin: float
    review_band: float
    pd_bands: list[dict]

    @property
    def break_even_pd(self) -> float:
        return self.margin / (self.lgd + self.margin)

    @property
    def review_range(self) -> tuple[float, float]:
        # |PD x (LGD + margin) - margin| < band  <=>  PD within band / (LGD + margin) of the break-even PD
        half = self.review_band / (self.lgd + self.margin)
        return max(self.break_even_pd - half, 0.0), min(self.break_even_pd + half, 1.0)


def load_decision_config(path: Path = DECISION_CONFIG) -> DecisionConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return DecisionConfig(lgd=raw["lgd"], margin=raw["margin"], review_band=raw["review_band"], pd_bands=raw["pd_bands"])


def decide(pd_value, cfg: DecisionConfig):
    """approve below the review range, decline above it, review inside it. Works on a number or an array."""
    lo, hi = cfg.review_range
    p = np.asarray(pd_value, dtype=float)
    out = np.where(p < lo, "approve", np.where(p >= hi, "decline", "review"))
    return out.item() if out.ndim == 0 else out


def risk_band(pd_value: float, cfg: DecisionConfig) -> dict:
    for b in cfg.pd_bands:
        if pd_value < b["upper"]:
            return {"band": b["band"], "label": b["label"]}
    return {"band": cfg.pd_bands[-1]["band"], "label": cfg.pd_bands[-1]["label"]}


def realised_cost(y, pd_values, amount, threshold: float, lgd: float, margin: float) -> float:
    """Actual cost on labelled data of approving everyone with PD below the threshold.

    Approved defaulters cost LGD x amount; declined good applicants cost margin x amount.
    """
    y = np.asarray(y)
    approved = np.asarray(pd_values) < threshold
    amount = np.asarray(amount, dtype=float)
    return float(np.sum(approved * y * lgd * amount) + np.sum(~approved * (1 - y) * margin * amount))


def expected_cost(pd_values, amount, threshold: float, lgd: float, margin: float) -> float:
    """The same cost as the model expects it, using PD in place of the actual outcome."""
    p = np.asarray(pd_values, dtype=float)
    approved = p < threshold
    amount = np.asarray(amount, dtype=float)
    return float(np.sum(approved * p * lgd * amount) + np.sum(~approved * (1 - p) * margin * amount))
