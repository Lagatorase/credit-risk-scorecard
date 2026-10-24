"""The final model: tuned LightGBM trained on the whole development set, with a Platt calibrator.

The calibrator is fit on out-of-fold predictions from the CV run of the chosen parameters,
never on predictions for rows the booster was trained on, because in-sample scores are
overconfident and would bend the calibration.
"""

from __future__ import annotations

import time
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


def train_like_final(X: pd.DataFrame, y: np.ndarray, params: dict, folds, meta: dict | None = None) -> tuple[FinalModel, np.ndarray]:
    """The final model's recipe for any feature set: Platt fit out of fold, then the booster on every row.

    The whole development set is 25% bigger than a training fold, so the tree count goes up by the same
    amount, as in the tuning step. Returns the model and its calibrated out-of-fold PDs. The fairness
    comparisons use this so they differ from the final model only in their inputs.
    """
    calibrator, oof = fit_platt_on_oof(lambda: LGBMModel(params=params), X, y, folds)
    final_params = {**params, "n_estimators": int(round(params["n_estimators"] * 1.25))}
    start = time.perf_counter()
    model = LGBMModel(params=final_params).fit(X, y)
    out = FinalModel(model=model, calibrator=calibrator, features=list(X.columns), meta={
        **(meta or {}),
        "params": final_params,
        "train_seconds": time.perf_counter() - start,
        "platt_coef": float(calibrator.lr.coef_[0][0]),
        "platt_intercept": float(calibrator.lr.intercept_[0]),
    })
    return out, calibrator.predict(oof)
