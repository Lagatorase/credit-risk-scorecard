"""SHAP explanations and full scoring results for single applicants.

SHAP values come from shap.TreeExplainer on the LightGBM booster, so they are in log-odds of the
raw (uncalibrated) model. Platt scaling is monotonic, so a factor that raises the raw log-odds
also raises the calibrated PD.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from .decision import DecisionConfig, decide, load_decision_config, risk_band
from .final_model import FinalModel
from .reasons import ReasonCoder


class Explainer:
    def __init__(self, model: FinalModel, reference: dict, decision_cfg: DecisionConfig | None = None,
                 reason_config: dict | None = None):
        self.model = model
        self.tree = shap.TreeExplainer(model.model.model_)
        self.cfg = decision_cfg or load_decision_config()
        self.coder = ReasonCoder(reference, reason_config)

    @classmethod
    def from_files(cls, models_dir: Path) -> "Explainer":
        # the reference distribution is saved next to the model by scripts/build_reference.py
        return cls(FinalModel.load(models_dir / "final_model.joblib"), joblib.load(models_dir / "reference.joblib"))

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        values = self.tree.shap_values(X[self.model.features])
        # older shap versions return one array per class for classifiers; I want the default class
        if isinstance(values, list):
            values = values[1]
        return np.asarray(values)

    @property
    def base_value(self) -> float:
        base = self.tree.expected_value
        return float(base[1] if np.ndim(base) else base)

    def explain(self, X: pd.DataFrame, k: int = 4) -> list[dict]:
        """PD, risk band, decision and both sets of reason codes for every row of X."""
        pds = self.model.predict_pd(X)
        values = self.shap_values(X)
        out = []
        for i in range(len(X)):
            row = X.iloc[i]
            out.append({
                "pd": float(pds[i]),
                "risk_band": risk_band(float(pds[i]), self.cfg),
                "decision": str(decide(float(pds[i]), self.cfg)),
                "reasons_officer": self.coder.top(values[i], row, self.model.features, k=k, audience="officer"),
                "reasons_customer": self.coder.top(values[i], row, self.model.features, k=k, audience="customer"),
                "shap": dict(zip(self.model.features, map(float, values[i]))),
                "base_value": self.base_value,
            })
        return out
