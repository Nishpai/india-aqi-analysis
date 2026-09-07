from __future__ import annotations

import json

import pandas as pd

from src.config import DATA_PROCESSED, OUT_REPORTS


def build_fused(cpcb: pd.DataFrame, hourly_daily: pd.DataFrame, climate: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily_cpcb = cpcb.copy()
    daily_fused = hourly_daily.merge(cpcb, on=["city", "date"], how="inner", suffixes=("", "_cpcb"))
    if "region_cpcb" in daily_fused.columns:
        daily_fused = daily_fused.drop(columns=["region_cpcb"])
    climate_keep = [c for c in climate.columns if c not in {"state"}]
    daily_fused = daily_fused.merge(
        climate[climate_keep],
        on=["city", "date"],
        how="left",
        suffixes=("", "_clim"),
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    daily_cpcb.to_parquet(DATA_PROCESSED / "daily_cpcb_wide.parquet", index=False)
    hourly_daily.to_parquet(DATA_PROCESSED / "hourly_daily.parquet", index=False)
    daily_fused.to_parquet(DATA_PROCESSED / "daily_fused.parquet", index=False)
    climate.to_parquet(DATA_PROCESSED / "daily_climate.parquet", index=False)
    return {
        "daily_cpcb_wide": daily_cpcb,
        "hourly_daily": hourly_daily,
        "daily_fused": daily_fused,
        "daily_climate": climate,
    }


def overlap_report(cpcb: pd.DataFrame, hourly_daily: pd.DataFrame, climate: pd.DataFrame, fused: pd.DataFrame) -> pd.DataFrame:
    def span(df: pd.DataFrame) -> tuple:
        return str(df["date"].min().date()), str(df["date"].max().date()), int(df["city"].nunique()), int(len(df))

    rows = [
        ["cpcb", *span(cpcb)],
        ["hourly_daily", *span(hourly_daily)],
        ["climate", *span(climate)],
        ["daily_fused", *span(fused)],
    ]
    table = pd.DataFrame(rows, columns=["table", "start", "end", "n_cities", "n_rows"])
    cpcb_cities = set(cpcb["city"].unique())
    hourly_cities = set(hourly_daily["city"].unique())
    climate_cities = set(climate["city"].unique())
    overlap = {
        "cpcb_and_hourly": sorted(cpcb_cities & hourly_cities),
        "all_three": sorted(cpcb_cities & hourly_cities & climate_cities),
        "cpcb_only": sorted(cpcb_cities - hourly_cities),
        "hourly_only": sorted(hourly_cities - cpcb_cities),
    }
    table.to_csv(OUT_REPORTS / "table_spans.csv", index=False)
    (OUT_REPORTS / "city_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    return table
