from __future__ import annotations

import pandas as pd

from src.config import SEASON_MONTHS, TRAIN_END, VALID_END


def add_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["dow"] = out[date_col].dt.dayofweek
    out["dayofyear"] = out[date_col].dt.dayofyear
    if "season" not in out.columns or out["season"].isna().all():
        out["season"] = out["month"].map(SEASON_MONTHS)
    return out


def add_lags(df: pd.DataFrame, group: str, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.sort_values([group, "date"]).copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(group)[col].shift(lag)
        out[f"{col}_roll7"] = out.groupby(group)[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    return out


def time_split(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    train_end = pd.Timestamp(TRAIN_END)
    valid_end = pd.Timestamp(VALID_END)
    dmin, dmax = out[date_col].min(), out[date_col].max()
    # Fall back to 70/15/15 if the configured years are not in range.
    if dmax <= train_end:
        q70 = dmin + 0.70 * (dmax - dmin)
        q85 = dmin + 0.85 * (dmax - dmin)
        out["split"] = "train"
        out.loc[out[date_col] > q70, "split"] = "valid"
        out.loc[out[date_col] > q85, "split"] = "test"
        return out
    out["split"] = "train"
    out.loc[out[date_col] > train_end, "split"] = "valid"
    out.loc[out[date_col] > valid_end, "split"] = "test"
    return out


def primary_pollutant(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return "Unknown"
    text = str(raw).upper().replace(" ", "").replace(".", "")
    text = text.replace("PARTICULATEMATTER", "PM")
    order = ["PM25", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "NH3"]
    compact = str(raw).upper().replace(" ", "")
    for token in ("PM2.5", "PM25", "PM10", "NO2", "SO2", "O3", "CO", "NH3"):
        needle = token.replace(".", "")
        if needle in compact.replace(".", "") or token in compact:
            if token in {"PM2.5", "PM25"}:
                return "PM2.5"
            return token if token != "PM25" else "PM2.5"
    if "PM2" in compact:
        return "PM2.5"
    if "PM10" in compact:
        return "PM10"
    first = str(raw).split(",")[0].strip()
    return first or "Unknown"
