from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, OUT_REPORTS
from src.ensure_data import ensure_raw_data
from src.joins import build_fused, overlap_report
from src.loaders import hourly_to_daily, load_climate, load_cpcb, load_hourly
from src.profile import write_profile


def main() -> None:
    info = ensure_raw_data()
    print("data mode:", info.get("mode"))
    cpcb = load_cpcb()
    climate = load_climate()
    hourly = load_hourly()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(DATA_PROCESSED / "hourly_29.parquet", index=False)
    hourly_daily = hourly_to_daily(hourly)
    tables = build_fused(cpcb, hourly_daily, climate)
    spans = overlap_report(cpcb, hourly_daily, climate, tables["daily_fused"])
    print(spans.to_string(index=False))
    write_profile(
        cpcb,
        hourly,
        climate,
        tables["daily_fused"],
        extra={"data_mode": info.get("mode"), "raw_info": info},
    )
    print("wrote", OUT_REPORTS / "phase1_profile.json")


if __name__ == "__main__":
    main()
