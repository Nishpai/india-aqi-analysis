from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import columns as C
from src.city_map import normalize_city, region_for
from src.config import (
    CLIMATE_GLOBS,
    CPCB_GLOBS,
    DATA_RAW,
    HOURLY_GLOBS,
    SEASON_MONTHS,
    indian_aqi_category,
)


def _first_match(globs: tuple[str, ...]) -> Path | None:
    hits: list[Path] = []
    for pattern in globs:
        hits.extend(DATA_RAW.glob(pattern))
    hits = [p for p in hits if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_size, reverse=True)
    return hits[0]


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False, utc=False)


def _clip_aqi(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0, upper=500)


def load_cpcb(path: Path | None = None) -> pd.DataFrame:
    path = path or _first_match(CPCB_GLOBS)
    if path is None:
        raise FileNotFoundError(f"No CPCB bulletin CSV in {DATA_RAW}")
    df = pd.read_csv(path, low_memory=False)
    date_col = C.require(df, "date")
    city_col = C.require(df, "city")
    aqi_col = C.require(df, "aqi")
    out = pd.DataFrame(
        {
            "date": parse_dates(df[date_col]).dt.normalize(),
            "city": df[city_col].map(normalize_city),
            "cpcb_aqi": _clip_aqi(df[aqi_col]),
        }
    )
    cat_col = C.resolve(df, "aqi_category")
    if cat_col:
        out["cpcb_category_raw"] = df[cat_col].astype(str)
    else:
        out["cpcb_category_raw"] = pd.NA
    pol_col = C.resolve(df, "prominent_pollutant")
    out["prominent_pollutant"] = df[pol_col].astype(str) if pol_col else pd.NA
    out = out.dropna(subset=["date", "city"])
    out["cpcb_category"] = out["cpcb_aqi"].map(indian_aqi_category)
    out["region"] = out["city"].map(region_for)
    out["source_file"] = path.name
    # One row per city-day; keep median AQI if bulletin duplicates exist.
    agg = (
        out.sort_values("cpcb_aqi")
        .groupby(["city", "date"], as_index=False)
        .agg(
            cpcb_aqi=("cpcb_aqi", "median"),
            cpcb_category_raw=("cpcb_category_raw", "last"),
            prominent_pollutant=("prominent_pollutant", "last"),
            region=("region", "first"),
            source_file=("source_file", "first"),
        )
    )
    agg["cpcb_category"] = agg["cpcb_aqi"].map(indian_aqi_category)
    return agg


def load_climate(path: Path | None = None) -> pd.DataFrame:
    path = path or _first_match(CLIMATE_GLOBS)
    if path is None:
        raise FileNotFoundError(f"No climate CSV in {DATA_RAW}")
    df = pd.read_csv(path, low_memory=False)
    date_col = C.require(df, "date")
    city_col = C.require(df, "city")
    out = pd.DataFrame(
        {
            "date": parse_dates(df[date_col]).dt.normalize(),
            "city": df[city_col].map(normalize_city),
        }
    )
    mapping = {
        "state": "state",
        "temp_avg": "temp_avg",
        "temp_max": "temp_max",
        "temp_min": "temp_min",
        "humidity": "humidity",
        "rainfall": "rainfall",
        "wind_speed": "wind_speed",
        "pressure": "pressure",
        "cloud_cover": "cloud_cover",
        "aqi": "climate_aqi",
        "aqi_category": "climate_aqi_category",
    }
    for logical, dest in mapping.items():
        src = C.resolve(df, logical)
        if src is None:
            continue
        if dest in {"climate_aqi"}:
            out[dest] = _clip_aqi(df[src])
        elif dest in {"temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "pressure", "cloud_cover"}:
            out[dest] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dest] = df[src]
    out = out.dropna(subset=["date", "city"])
    return out.drop_duplicates(["city", "date"], keep="last")


def load_hourly(path: Path | None = None) -> pd.DataFrame:
    path = path or _first_match(HOURLY_GLOBS)
    if path is None:
        raise FileNotFoundError(f"No hourly AQI CSV in {DATA_RAW}")
    df = pd.read_csv(path, low_memory=False)
    dt_col = C.resolve(df, "datetime") or C.require(df, "date")
    city_col = C.require(df, "city")
    out = pd.DataFrame(
        {
            "datetime": parse_dates(df[dt_col]),
            "city": df[city_col].map(normalize_city),
        }
    )
    numeric = {
        "temp_avg": "temp",
        "humidity": "humidity",
        "rainfall": "rain",
        "wind_speed": "wind_speed",
        "wind_dir": "wind_dir",
        "pressure": "pressure",
        "cloud_cover": "cloud_cover",
        "pm25": "pm25",
        "pm10": "pm10",
        "co": "co",
        "no2": "no2",
        "so2": "so2",
        "o3": "o3",
        "us_aqi": "us_aqi",
        "eu_aqi": "eu_aqi",
        "lat": "lat",
        "lon": "lon",
    }
    for logical, dest in numeric.items():
        src = C.resolve(df, logical)
        if src is not None:
            out[dest] = pd.to_numeric(df[src], errors="coerce")
    for logical, dest in (
        ("season", "season"),
        ("state", "state"),
        ("aqi_category", "us_aqi_category"),
    ):
        src = C.resolve(df, logical)
        if src is not None:
            out[dest] = df[src]
    for logical, dest in (
        ("festival", "festival"),
        ("crop_burning", "crop_burning"),
        ("temp_inversion", "temp_inversion"),
        ("wind_stagnation", "wind_stagnation"),
    ):
        src = C.resolve(df, logical)
        if src is not None:
            out[dest] = _as_bool(df[src])
    out = out.dropna(subset=["datetime", "city"])
    out["date"] = out["datetime"].dt.normalize()
    if "season" not in out.columns:
        out["season"] = out["datetime"].dt.month.map(SEASON_MONTHS)
    out["region"] = out["city"].map(region_for)
    out["source_file"] = path.name
    if "us_aqi" in out.columns:
        out["us_aqi"] = out["us_aqi"].clip(lower=0, upper=500)
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    mapped = series.astype(str).str.strip().str.lower()
    return mapped.isin({"1", "true", "yes", "y", "t"})


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    flags = [c for c in ("festival", "crop_burning", "temp_inversion", "wind_stagnation") if c in hourly.columns]
    means = [c for c in ("temp", "humidity", "wind_speed", "pressure", "cloud_cover", "pm25", "pm10", "co", "no2", "so2", "o3", "us_aqi", "eu_aqi") if c in hourly.columns]
    maxs = [c for c in ("pm25", "pm10", "us_aqi") if c in hourly.columns]
    sums = [c for c in ("rain",) if c in hourly.columns]

    grouped = hourly.groupby(["city", "date"], sort=False)
    parts = []
    if means:
        m = grouped[means].mean()
        m.columns = [f"{c}_mean" for c in m.columns]
        parts.append(m)
    if maxs:
        x = grouped[maxs].max()
        x.columns = [f"{c}_max" for c in x.columns]
        parts.append(x)
    if sums:
        s = grouped[sums].sum()
        s.columns = [f"{c}_sum" for c in s.columns]
        parts.append(s)
    if flags:
        f = grouped[flags].max().astype(float)
        parts.append(f)
    meta = grouped.agg(n_hours=("datetime", "count"), region=("region", "first"))
    if "season" in hourly.columns:
        meta["season"] = grouped["season"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    else:
        meta["season"] = pd.NaT
    parts.append(meta)
    daily = pd.concat(parts, axis=1).reset_index()
    daily["season"] = daily["date"].dt.month.map(SEASON_MONTHS)
    daily["region"] = daily["city"].map(region_for)
    return daily
