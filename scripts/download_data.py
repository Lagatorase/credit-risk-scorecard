"""Downloads the Home Credit Default Risk competition files I use into data/raw.

  python scripts/download_data.py

Needs a Kaggle API token and the competition rules accepted on the Kaggle website.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# makes "import scorecard" work when I run this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorecard.config import get_settings  # noqa: E402

COMPETITION = "home-credit-default-risk"

# the seven tables I use; application_test has no labels, so I skip it
FILES = [
    "application_train.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]

CREDENTIALS_HELP = """
Kaggle credentials are not set up. Either of these fixes it:
  A. Log in through the browser (easiest):
       .venv\\Scripts\\kaggle auth login
  B. Use an API token:
       1. Open https://www.kaggle.com/settings/api and click "Generate New Token".
       2. Save the token text to C:\\Users\\<you>\\.kaggle\\access_token
          (or set the KAGGLE_API_TOKEN environment variable).
Then run this script again.
"""

RULES_HELP = f"""
Kaggle refused the download (403). You need to accept the competition rules:
  1. Open https://www.kaggle.com/competitions/{COMPETITION}/rules while signed in.
  2. Click "I Understand and Accept" (you may need to verify your phone number first).
  3. Run this script again.
"""


def authenticate():
    # the kaggle package reads credentials when it is imported, so I import it inside a try
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except (OSError, ValueError, SystemExit, Exception) as exc:  # the package raises a few different types
        print(f"Authentication failed: {exc}")
        print(CREDENTIALS_HELP)
        sys.exit(1)


def download(api, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw_dir / name
        if target.exists():
            print(f"{name}: already downloaded")
            continue
        print(f"{name}: downloading...")
        try:
            api.competition_download_file(COMPETITION, name, path=str(raw_dir), quiet=True)
        except Exception as exc:
            # a 403 here almost always means the rules haven't been accepted
            if "403" in str(exc) or "Forbidden" in str(exc):
                print(RULES_HELP)
            else:
                print(f"Download failed: {exc}")
            sys.exit(1)
        # Kaggle serves each file zipped, so I unzip it and remove the zip
        zipped = raw_dir / f"{name}.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(raw_dir)
            zipped.unlink()
        print(f"{name}: {target.stat().st_size / 1e6:,.0f} MB")


if __name__ == "__main__":
    settings = get_settings()
    download(authenticate(), settings.raw_dir)
    print("Done. Next: python scripts/convert_to_parquet.py")
