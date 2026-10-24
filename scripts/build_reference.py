"""Saves where each feature's values sit among development applicants, for the reason code wording.

  python scripts/build_reference.py

Uses the development set only, so a hold-out applicant is never compared against itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from scorecard.config import ROOT, get_settings  # noqa: E402
from scorecard.reasons import build_reference, save_reference  # noqa: E402
from scorecard.splits import load_splits, split_frame  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    features = (ROOT / "config" / "model_features.txt").read_text().split()
    df = pd.read_parquet(settings.processed_dir / "features.parquet")
    dev, _ = split_frame(df, load_splits())
    ref = build_reference(dev[features])
    save_reference(ref, settings.models_dir / "reference.joblib")
    print(f"Saved reference distributions for {len(ref)} features from {len(dev):,} development applicants")
