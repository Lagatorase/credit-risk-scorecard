"""The final model: tuned LightGBM trained on the whole development set, with a Platt calibrator.

The calibrator is fit on out-of-fold predictions from the CV run of the chosen parameters,
never on predictions for rows the booster was trained on, because in-sample scores are
overconfident and would bend the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .models import LGBMModel, _Platt
from .splits import run_cv


def fit_platt_on_oof(make_model, X: pd.DataFrame, y: np.ndarray, folds) -> tuple[_Platt, np.ndarray]:
    """Runs CV, then fits Platt scaling on the out-of-fold raw scores only.

    Every score the calibrator sees comes from a fold model that was not trained on that row.
    """
    res = run_cv(make_model, X, y, folds)
    oof = res["oof"]
    # run_cv fills every development row exactly once, so a NaN here would mean a row was never scored out of fold
    if np.isnan(oof).any():
        raise ValueError("some rows have no out-of-fold score")
    return _Platt().fit(oof, y), oof


@dataclass
class FinalModel:
    model: LGBMModel
    calibrator: _Platt
    features: list[str]
    meta: dict = field(default_factory=dict)

    def raw_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        # Platt is monotonic, so it changes the probabilities without changing the ranking
        return self.calibrator.predict(self.raw_pd(X))

    @property
    def booster(self):
        return self.model.model_.booster_

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FinalModel":
        return joblib.load(path)
