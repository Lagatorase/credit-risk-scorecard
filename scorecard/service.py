"""Scoring for one applicant at a time, shared by the API and the Streamlit app.

Takes the applicant's model inputs as a dict, checks them, and returns the PD, risk band, decision,
reason codes for both readers and the written explanations. Also holds the what-if helper that
recomputes the ratio features when an officer changes the loan amount, income or years employed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Settings
from .explain import Explainer
from .features import add_derived_ratios
from .llm import LLM, get_llm
from .narrative import explain_both

MODEL_CARD = ROOT / "docs" / "model_card.md"

# the raw inputs an officer can change in the what-if panel, with display labels
EDITABLE = {
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Yearly repayment",
    "AMT_INCOME_TOTAL": "Yearly income",
    "AMT_GOODS_PRICE": "Price of the goods",
    "EMPLOYED_YEARS": "Years in current job",
}


class InputError(ValueError):
    """An applicant record the model can't score, with a message that says why."""


def apply_whatif(row: pd.Series, changes: dict) -> pd.Series:
    """A copy of the applicant with some raw inputs changed and every ratio built from them recomputed."""
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise InputError(f"these inputs can't be changed in the what-if: {sorted(unknown)}")
    df = pd.DataFrame([row.to_dict()])
    for name, value in changes.items():
        df[name] = float(value)
    if "EMPLOYED_YEARS" in changes:
        # a value for years employed means this applicant now has a current job
        df["DAYS_EMPLOYED_PLACEHOLDER"] = 0.0
    add_derived_ratios(df)
    return df.iloc[0]


@dataclass
class ScoringService:
    explainer: Explainer
    llm: LLM | None
    customer_llm: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "ScoringService":
        return cls(Explainer.from_files(settings.models_dir), get_llm(settings),
                   customer_llm=settings.customer_explanation == "llm")

    @property
    def features(self) -> list[str]:
        return self.explainer.model.features

    def _categorical(self, feature: str) -> bool:
        return self.explainer.coder.reference[feature]["kind"] == "category"

    def to_frame(self, applicant: dict) -> pd.DataFrame:
        """One-row frame of model inputs; missing inputs become NaN, which LightGBM handles as it did in training."""
        unknown = set(applicant) - set(self.features)
        if unknown:
            raise InputError(f"unknown inputs: {sorted(unknown)[:10]}")
        row = {}
        for f in self.features:
            value = applicant.get(f)
            if self._categorical(f):
                row[f] = pd.Categorical([None if value is None or pd.isna(value) else str(value)])
            else:
                try:
                    row[f] = [np.nan if value is None else float(value)]
                except (TypeError, ValueError) as exc:
                    raise InputError(f"{f} must be a number, got {value!r}") from exc
        return pd.DataFrame(row)

    def score(self, applicant: dict | pd.Series, explanations: bool = True, include_shap: bool = False) -> dict:
        """PD, band, decision, both reason lists and, if asked, the written explanations for one applicant."""
        if isinstance(applicant, pd.Series):
            applicant = {k: (None if pd.isna(v) else v) for k, v in applicant.items() if k in self.features}
        X = self.to_frame(applicant)
        ex = self.explainer.explain(X)[0]
        out = {
            "pd": ex["pd"],
            "risk_band": ex["risk_band"],
            "decision": ex["decision"],
            "reasons_officer": ex["reasons_officer"],
            "reasons_customer": ex["reasons_customer"],
        }
        if explanations:
            texts = explain_both(self.llm, ex, customer_llm=self.customer_llm)
            out["explanations"] = {aud: {k: t[k] for k in ("text", "label", "source")} for aud, t in texts.items()}
        if include_shap:
            out["shap"] = ex["shap"]
            out["base_value"] = ex["base_value"]
        return out


def model_card_text(path: Path = MODEL_CARD) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "Model card not found."
