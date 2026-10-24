"""Saves a small set of hold-out applicants for the Streamlit demo.

  python scripts/export_demo_applicants.py

Writes models/demo_applicants.parquet: the three applicants used in the reports plus seven per decision
drawn at random (seed 7). It sits in models/, which isn't committed, because the rows come from Home
Credit's data; anyone running the demo builds it from their own copy of the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from scorecard.config import get_settings  # noqa: E402
from scorecard.decision import decide, load_decision_config  # noqa: E402

# the applicants from the reason code and explanation reports come first, so the demo matches the docs
REPORT_IDS = [140957, 290683, 169973]
PER_DECISION = 7

if __name__ == "__main__":
    settings = get_settings()
    features = (Path(__file__).resolve().parent.parent / "config" / "model_features.txt").read_text().split()
    holdout = pd.read_parquet(settings.reports_dir / "cache" / "holdout_pd.parquet")
    holdout = holdout.assign(decision=decide(holdout["pd"].to_numpy(), load_decision_config()))
    rest = holdout[~holdout["SK_ID_CURR"].isin(REPORT_IDS)]
    picked = pd.concat([rest[rest["decision"] == d].sample(PER_DECISION, random_state=7)
                        for d in ("approve", "review", "decline")])
    ids = REPORT_IDS + picked["SK_ID_CURR"].tolist()
    rows = pd.read_parquet(settings.processed_dir / "features.parquet", columns=["SK_ID_CURR", *features])
    demo = rows.set_index("SK_ID_CURR").loc[ids].reset_index()
    demo.to_parquet(settings.models_dir / "demo_applicants.parquet")
    print(f"Saved {len(demo)} demo applicants to {settings.models_dir / 'demo_applicants.parquet'}")
