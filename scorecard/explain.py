"""SHAP explanations and full scoring results for single applicants.

SHAP values come from shap.TreeExplainer on the LightGBM booster, so they are in log-odds of the
raw (uncalibrated) model. Platt scaling is monotonic, so a factor that raises the raw log-odds
also raises the calibrated PD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .decision import DecisionConfig, decide, load_decision_config, risk_band
from .final_model import FinalModel
from .reasons import load_reason_map, top_reasons


class Explainer:
    def __init__(self, model: FinalModel, decision_cfg: DecisionConfig | None = None, reason_map: dict | None = None):
        self.model = model
        self.tree = shap.TreeExplainer(model.model.model_)
        self.cfg = decision_cfg or load_decision_config()
        self.mapping = reason_map or load_reason_map()

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
            args = (values[i], row, self.model.features, self.model.medians, self.mapping)
            out.append({
                "pd": float(pds[i]),
                "risk_band": risk_band(float(pds[i]), self.cfg),
                "decision": str(decide(float(pds[i]), self.cfg)),
                "reasons_officer": top_reasons(*args, k=k, audience="officer"),
                "reasons_customer": top_reasons(*args, k=k, audience="customer"),
                "shap": dict(zip(self.model.features, map(float, values[i]))),
                "base_value": self.base_value,
            })
        return out
