"""Builds the feature table once and saves it to data/processed/features.parquet.

  python scripts/build_features.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorecard.config import get_settings  # noqa: E402
from scorecard.features import build_features, feature_groups  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    start = time.perf_counter()
    df = build_features(settings.processed_dir)
    out = settings.processed_dir / "features.parquet"
    df.to_parquet(out, index=False)

    # a short summary so I can see each group's size and the memory footprint
    print(f"\n{len(df):,} applicants, {df.shape[1] - 2} features, "
          f"{df.memory_usage(deep=True).sum() / 1e6:,.0f} MB in memory, built in {time.perf_counter() - start:.0f}s")
    for name, cols in feature_groups(df.columns).items():
        print(f"  {name}: {len(cols)}")
