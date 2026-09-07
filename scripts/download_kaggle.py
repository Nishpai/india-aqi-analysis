from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

DATASETS = [
    "ankushnarwade/indian-climate-dataset-20242025",
    "bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025",
    "krishnandansha/aqi-india-2020-to-2025-october",
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("Missing %s" % cred)
        print("Create a Kaggle API token (Account → API) and save it as kaggle.json, then rerun.")
        sys.exit(1)
    for slug in DATASETS:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(RAW), "--unzip"]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
    print(json.dumps({"raw": str(RAW), "files": [p.name for p in RAW.iterdir() if p.is_file()]}, indent=2))


if __name__ == "__main__":
    main()
