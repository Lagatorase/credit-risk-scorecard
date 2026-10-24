"""Makes the fixed 80/20 development/hold-out split once and saves it to data/splits.json.

  python scripts/make_splits.py

Refuses to overwrite an existing split, because every result depends on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorecard.config import get_settings  # noqa: E402
from scorecard.data import load_table  # noqa: E402
from scorecard.splits import SPLITS_PATH, make_splits, save_splits  # noqa: E402

if __name__ == "__main__":
    if SPLITS_PATH.exists():
        raise SystemExit(f"{SPLITS_PATH} already exists. The split is fixed, so I won't overwrite it.")
    app = load_table(get_settings().processed_dir, "application_train", columns=["SK_ID_CURR", "TARGET"])
    splits = make_splits(app["SK_ID_CURR"].to_numpy(), app["TARGET"].to_numpy())
    save_splits(splits)

    # I print the default rate in each part to show the stratification worked
    is_test = app["SK_ID_CURR"].isin(set(splits["test_ids"]))
    print(f"Development: {splits['n_dev']:,} applicants, default rate {app.loc[~is_test, 'TARGET'].mean():.3%}")
    print(f"Hold-out:    {splits['n_test']:,} applicants, default rate {app.loc[is_test, 'TARGET'].mean():.3%}")
    print(f"Saved to {SPLITS_PATH} ({SPLITS_PATH.stat().st_size / 1e3:,.0f} KB)")
