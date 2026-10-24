"""The fixed development/hold-out split and the 5-fold CV used for every model comparison.

The split is made once by scripts/make_splits.py and saved to data/splits.json, which is committed.
I never regenerate it, so every result in the reports refers to the same rows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ROOT, SEED

SPLITS_PATH = ROOT / "data" / "splits.json"
N_FOLDS = 5


def make_splits(ids: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = SEED) -> dict:
    """Stratified 80/20 split of applicant IDs, so both parts keep the same default rate."""
    dev_ids, test_ids = train_test_split(ids, test_size=test_size, stratify=y, random_state=seed)
    # I only store the hold-out IDs: the development set is every other labelled applicant,
    # which keeps the committed file small
    return {
        "seed": seed,
        "test_size": test_size,
        "n_dev": int(len(dev_ids)),
        "n_test": int(len(test_ids)),
        "test_ids": sorted(int(i) for i in test_ids),
    }


def save_splits(splits: dict, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # compact separators keep the file around 430 KB for 61,503 IDs
    path.write_text(json.dumps(splits, separators=(",", ":")))


def load_splits(path: Path = SPLITS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing, run scripts/make_splits.py once")
    return json.loads(path.read_text())


def split_frame(df: pd.DataFrame, splits: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a frame with SK_ID_CURR into development and hold-out parts, both sorted by ID."""
    is_test = df["SK_ID_CURR"].isin(set(splits["test_ids"]))
    dev = df[~is_test].sort_values("SK_ID_CURR").reset_index(drop=True)
    test = df[is_test].sort_values("SK_ID_CURR").reset_index(drop=True)
    # a quick guard so a stale features file can't silently change the split sizes
    if len(test) != splits["n_test"] or len(dev) != splits["n_dev"]:
        raise ValueError(f"split sizes don't match splits.json: dev {len(dev)} vs {splits['n_dev']}, "
                         f"test {len(test)} vs {splits['n_test']}")
    return dev, test


def cv_folds(y: np.ndarray, n_splits: int = N_FOLDS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds over the development set. Rows must be sorted by ID for the folds to be repeatable."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def run_cv(make_model: Callable[[], object], X: pd.DataFrame, y: np.ndarray,
           folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Fits a fresh model on each training fold and predicts the matching validation fold.

    The model object owns all of its preprocessing (binning, encoding, calibration), so anything
    it learns comes from the training fold only. Returns out-of-fold predictions and timings.
    """
    oof = np.full(len(y), np.nan)
    fold_preds, fit_seconds = [], []
    for train_idx, val_idx in folds:
        model = make_model()
        start = time.perf_counter()
        model.fit(X.iloc[train_idx], y[train_idx])
        fit_seconds.append(time.perf_counter() - start)
        # I keep each fold's predictions separately too, so metrics can be averaged per fold
        p = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = p
        fold_preds.append((val_idx, p))
    return {"oof": oof, "fold_preds": fold_preds, "fit_seconds": fit_seconds}
