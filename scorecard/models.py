"""LightGBM models used in the experiments: plain, tuned with Optuna, and tuned plus calibrated.

Each model has fit(X, y) and predict_proba(X), and does all of its own tuning and calibration
inside fit, so when run_cv gives it one training fold nothing leaks from the validation fold.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import SEED

# six physical cores on my laptop; LightGBM is slower with hyper-threads than without
N_JOBS = 6

BASE_PARAMS = {"random_state": SEED, "n_jobs": N_JOBS, "verbose": -1}


class LGBMModel:
    """LightGBM with its default parameters (100 trees, learning rate 0.1, 31 leaves) unless told otherwise."""

    def __init__(self, params: dict | None = None, class_weight: str | None = None):
        self.params = params or {}
        self.class_weight = class_weight

    def fit(self, X, y):
        self.model_ = lgb.LGBMClassifier(**BASE_PARAMS, class_weight=self.class_weight, **self.params)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importance_(self) -> pd.Series:
        # total gain is a better importance measure than split counts
        booster = self.model_.booster_
        return pd.Series(booster.feature_importance("gain"), index=booster.feature_name())


def _fingerprint(X: pd.DataFrame) -> str:
    # identifies a training fold by its rows and columns, so tuning results can be reused across experiments
    h = hashlib.sha1(np.ascontiguousarray(X.index.to_numpy()).tobytes())
    h.update(",".join(X.columns).encode())
    return h.hexdigest()[:16]


def _search_space(trial: optuna.Trial) -> dict:
    # ranges around what usually works for tabular credit data; learning rate stays at 0.03 or above to keep trials fast
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune_lgbm(X: pd.DataFrame, y, n_trials: int, cache_dir: Path | None = None, log=print) -> dict:
    """Tunes LightGBM on one training set with Optuna and returns the best parameters and tree count.

    The training set is split 80/20 again: trees are grown on the inner 80% with early stopping on the
    inner 20%, and Optuna maximises AUC on that inner 20%. The outer validation fold is never seen.
    Results are cached by fold fingerprint so the calibration experiment reuses them.
    """
    y = np.asarray(y)
    key = f"{_fingerprint(X)}_{n_trials}"
    cache_file = cache_dir / f"tuned_{key}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text())

    X_in, X_es, y_in, y_es = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    start = time.perf_counter()

    def objective(trial):
        params = _search_space(trial)
        model = lgb.LGBMClassifier(**BASE_PARAMS, n_estimators=3000, **params)
        model.fit(X_in, y_in, eval_X=X_es, eval_y=y_es, eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr("n_estimators", int(model.best_iteration_))
        return roc_auc_score(y_es, model.predict_proba(X_es)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    result = {
        "params": {**best.params, "subsample_freq": 1},
        # the final model trains on all of the training set, 25% more rows than the inner split,
        # so I scale the early-stopped tree count up by the same amount
        "n_estimators": int(round(best.user_attrs["n_estimators"] * 1.25)),
        "inner_auc": best.value,
        "n_trials": n_trials,
        "seconds": time.perf_counter() - start,
    }
    log(f"    tuned in {result['seconds']:.0f}s: inner AUC {best.value:.4f}, {result['n_estimators']} trees")
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2))
    return result


class TunedLGBM(LGBMModel):
    """LightGBM tuned with Optuna on each training fold separately (nested inside the CV)."""

    def __init__(self, n_trials: int = 30, cache_dir: Path | None = None):
        super().__init__()
        self.n_trials = n_trials
        self.cache_dir = cache_dir

    def fit(self, X, y):
        self.tuning_ = tune_lgbm(X, y, self.n_trials, self.cache_dir)
        self.params = {**self.tuning_["params"], "n_estimators": self.tuning_["n_estimators"]}
        return super().fit(X, y)


class _Platt:
    """Platt scaling: a logistic regression on the log-odds of the raw probability."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# raw out-of-fold predictions and final boosters, shared between the isotonic and Platt runs in one process
_CALIBRATION_MEMO: dict[str, tuple] = {}


class CalibratedLGBM(TunedLGBM):
    """The tuned model with isotonic or Platt calibration, fit on the training fold only.

    I split the training fold three ways, train a booster on two parts and predict the third, so
    every training row gets a prediction from a model that didn't see it. The calibrator is fit on
    those predictions, then applied to a booster trained on the whole training fold (the same idea
    as scikit-learn's CalibratedClassifierCV with ensemble=False).
    """

    def __init__(self, method: str = "isotonic", n_trials: int = 30, cache_dir: Path | None = None):
        super().__init__(n_trials=n_trials, cache_dir=cache_dir)
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be isotonic or platt")
        self.method = method

    def fit(self, X, y):
        y = np.asarray(y)
        key = f"{_fingerprint(X)}_{self.n_trials}"
        if key not in _CALIBRATION_MEMO:
            super().fit(X, y)
            oof = np.zeros(len(y))
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            for tr, va in skf.split(X, y):
                m = lgb.LGBMClassifier(**BASE_PARAMS, **self.params).fit(X.iloc[tr], y[tr])
                oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            _CALIBRATION_MEMO[key] = (self.model_, self.params, self.tuning_, oof)
        self.model_, self.params, self.tuning_, oof = _CALIBRATION_MEMO[key]
        self.calibrator_ = (IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1) if self.method == "isotonic"
                            else _Platt()).fit(oof, y)
        return self

    def predict_proba(self, X):
        p = self.calibrator_.predict(self.model_.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def clear_calibration_memo() -> None:
    _CALIBRATION_MEMO.clear()
