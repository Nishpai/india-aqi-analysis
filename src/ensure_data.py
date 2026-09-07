from __future__ import annotations

from src.config import CLIMATE_GLOBS, CPCB_GLOBS, HOURLY_GLOBS
from src.loaders import _first_match
from src.synthetic import write_synthetic_raw


def ensure_raw_data() -> dict:
    found = {
        "cpcb": _first_match(CPCB_GLOBS),
        "hourly": _first_match(HOURLY_GLOBS),
        "climate": _first_match(CLIMATE_GLOBS),
    }
    if all(found.values()):
        return {"mode": "kaggle_or_local", "paths": {k: str(v) for k, v in found.items()}}
    if any(found.values()):
        missing = [k for k, v in found.items() if v is None]
        present = {k: str(v) for k, v in found.items() if v is not None}
        raise FileNotFoundError(
            f"Partial raw data in data/raw (missing {missing}, found {present}). "
            "Add the rest via python scripts/download_kaggle.py, or empty data/raw to use synthetic CSVs."
        )
    written = write_synthetic_raw()
    written["mode"] = "synthetic"
    return written
