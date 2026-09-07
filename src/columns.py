"""Flexible column matching so Kaggle schema drift does not break the pipeline."""

from __future__ import annotations

import re

import pandas as pd

_STRIP_UNITS = re.compile(r"[\s_\-]*\([^)]*\)[\s_]*")


def clean_name(name: str) -> str:
    n = str(name).strip()
    n = _STRIP_UNITS.sub("", n)
    n = n.lower().replace("%", "pct")
    n = re.sub(r"[^\w]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    n = n.replace("pm2_5", "pm25").replace("pm_2_5", "pm25")
    n = n.replace("ugm3", "").replace("ug_m3", "").strip("_")
    return n


ALIASES: dict[str, tuple[str, ...]] = {
    "datetime": ("datetime", "date_time", "timestamp", "date"),
    "date": ("date", "datetime", "bulletin_date"),
    "city": ("city", "city_name", "location"),
    "state": ("state", "state_name"),
    "aqi": ("aqi", "aqi_value", "cpcb_aqi", "indian_aqi"),
    "aqi_category": ("aqi_category", "air_quality", "air_quality_status", "category"),
    "prominent_pollutant": ("prominent_pollutant", "prominent_pollutants", "pollutant"),
    "temp_avg": ("temperature_avg", "temp_avg", "temp_2m_c", "temperature", "temp"),
    "temp_max": ("temperature_max", "temp_max", "tmax"),
    "temp_min": ("temperature_min", "temp_min", "tmin"),
    "humidity": ("humidity", "humidity_percent", "humidity_pct", "rh"),
    "rainfall": ("rainfall", "rain_mm", "precipitation_mm", "precip", "rain"),
    "wind_speed": ("wind_speed", "wind_speed_10m_kmh", "windspeed"),
    "wind_dir": ("wind_dir_10m", "wind_dir", "wind_direction"),
    "pressure": ("pressure", "pressure_msl_hpa", "msl"),
    "cloud_cover": ("cloud_cover", "cloud_cover_percent", "cloud_cover_pct"),
    "pm25": ("pm25", "pm2_5", "pm2_5_ugm3", "pm25_ugm3"),
    "pm10": ("pm10", "pm10_ugm3"),
    "co": ("co", "co_ugm3"),
    "no2": ("no2", "no2_ugm3"),
    "so2": ("so2", "so2_ugm3"),
    "o3": ("o3", "o3_ugm3"),
    "us_aqi": ("us_aqi", "usa_aqi"),
    "eu_aqi": ("eu_aqi",),
    "season": ("season",),
    "festival": ("festival_period", "festival", "is_festival"),
    "crop_burning": ("crop_burning_season", "crop_burning"),
    "temp_inversion": ("temp_inversion", "temperature_inversion"),
    "wind_stagnation": ("wind_stagnation", "stagnation"),
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "lon", "lng"),
}


def index_columns(df: pd.DataFrame) -> dict[str, str]:
    return {clean_name(c): c for c in df.columns}


def resolve(df: pd.DataFrame, logical: str) -> str | None:
    idx = index_columns(df)
    if logical in idx:
        return idx[logical]
    for alias in ALIASES.get(logical, ()):
        if alias in idx:
            return idx[alias]
        collapsed = alias.replace("_", "")
        for key, orig in idx.items():
            if key.replace("_", "") == collapsed:
                return orig
    return None


def require(df: pd.DataFrame, logical: str) -> str:
    col = resolve(df, logical)
    if col is None:
        raise KeyError(f"Cannot find column for '{logical}'. Have: {list(df.columns)}")
    return col


def rename_known(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for logical, dest in mapping.items():
        src = resolve(out, logical)
        if src is not None and src != dest:
            rename[src] = dest
    return out.rename(columns=rename)
