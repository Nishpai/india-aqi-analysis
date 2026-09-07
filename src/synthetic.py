"""Vectorized synthetic CSVs so Phase 1–2 run before Kaggle downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.city_map import region_for
from src.config import DATA_RAW, SEASON_MONTHS

CITIES = [
    "Delhi",
    "Lucknow",
    "Jaipur",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Bengaluru",
    "Ahmedabad",
]

_BASE_AQI = {
    "Delhi": 180,
    "Lucknow": 165,
    "Jaipur": 140,
    "Mumbai": 95,
    "Kolkata": 130,
    "Chennai": 80,
    "Bengaluru": 75,
    "Ahmedabad": 115,
}

_COORDS = {
    "Delhi": (28.61, 77.21),
    "Lucknow": (26.85, 80.95),
    "Jaipur": (26.91, 75.79),
    "Mumbai": (19.08, 72.88),
    "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27),
    "Bengaluru": (12.97, 77.59),
    "Ahmedabad": (23.02, 72.57),
}


def _season_boost(month: np.ndarray, region: str) -> np.ndarray:
    winter = np.isin(month, [12, 1, 2]).astype(float)
    monsoon = np.isin(month, [6, 7, 8, 9]).astype(float)
    post = np.isin(month, [10, 11]).astype(float)
    if region == "IGP":
        return 90 * winter + 70 * post - 55 * monsoon
    if region == "Northwest":
        return 40 * winter + 35 * np.isin(month, [4, 5, 6]).astype(float) - 30 * monsoon
    if region == "West":
        return 20 * winter - 25 * monsoon
    if region == "East":
        return 45 * winter - 40 * monsoon
    return 10 * winter - 15 * monsoon


def _diwali_boost(dates: pd.DatetimeIndex) -> np.ndarray:
    # Approximate Diwali window: last week of October.
    return ((dates.month == 10) & (dates.day >= 20)).astype(float) * 80


def _primary_pollutant(month: int, region: str, aqi: float) -> str:
    if month in (3, 4, 5) and aqi > 100:
        return "O3" if region in {"South", "West"} else "PM10"
    if month in (10, 11) and region == "IGP":
        return "PM2.5"
    if month in (12, 1, 2) and region in {"IGP", "Northwest", "East"}:
        return "PM2.5"
    if aqi < 80:
        return "O3" if month in (4, 5) else "PM10"
    return "PM2.5"


def write_synthetic_raw(seed: int = 7) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    cpcb_days = pd.date_range("2020-01-01", "2025-10-31", freq="D")
    cpcb_rows = []
    for city in CITIES:
        region = region_for(city)
        months = cpcb_days.month.values
        aqi = (
            _BASE_AQI[city]
            + _season_boost(months, region)
            + _diwali_boost(cpcb_days)
            + rng.normal(0, 18, size=len(cpcb_days))
        )
        aqi = np.clip(aqi, 18, 480)
        cats = pd.cut(
            aqi,
            bins=[-1, 50, 100, 200, 300, 400, 1000],
            labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        )
        pols = [_primary_pollutant(int(m), region, float(a)) for m, a in zip(months, aqi)]
        cpcb_rows.append(
            pd.DataFrame(
                {
                    "Date": cpcb_days.strftime("%Y-%m-%d"),
                    "S.No": np.arange(1, len(cpcb_days) + 1),
                    "City": city,
                    "Air Quality": cats.astype(str),
                    "AQI Value": np.round(aqi, 1),
                    "Prominent Pollutant": pols,
                }
            )
        )
    cpcb = pd.concat(cpcb_rows, ignore_index=True)
    cpcb_path = DATA_RAW / "AQI_2020_to_2025_October.csv"
    cpcb.to_csv(cpcb_path, index=False)

    climate_days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    clim_rows = []
    for city in CITIES:
        region = region_for(city)
        months = climate_days.month.values
        temp = 18 + 10 * np.sin((months - 3) / 12 * 2 * np.pi) + rng.normal(0, 2, len(climate_days))
        if region == "South":
            temp += 6
        rain = np.where(np.isin(months, [6, 7, 8, 9]), rng.gamma(2, 4, len(climate_days)), rng.exponential(0.4, len(climate_days)))
        aqi = np.clip(
            _BASE_AQI[city] + _season_boost(months, region) + rng.normal(0, 16, len(climate_days)),
            20,
            450,
        )
        cats = pd.cut(aqi, bins=[-1, 50, 100, 200, 300, 400, 1000], labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
        clim_rows.append(
            pd.DataFrame(
                {
                    "Date": climate_days.strftime("%Y-%m-%d"),
                    "City": city,
                    "State": region,
                    "Temperature_Max": np.round(temp + 6, 1),
                    "Temperature_Min": np.round(temp - 5, 1),
                    "Temperature_Avg": np.round(temp, 1),
                    "Humidity": np.clip(55 + 20 * np.isin(months, [6, 7, 8, 9]) + rng.normal(0, 8, len(climate_days)), 20, 99),
                    "Rainfall": np.round(rain, 2),
                    "Wind_Speed": np.round(np.clip(rng.normal(10, 3, len(climate_days)), 1, 30), 1),
                    "AQI": np.round(aqi, 1),
                    "AQI_Category": cats.astype(str),
                    "Pressure": np.round(rng.normal(1008, 4, len(climate_days)), 1),
                    "Cloud_Cover": np.clip(rng.normal(40, 20, len(climate_days)), 0, 100),
                }
            )
        )
    climate = pd.concat(clim_rows, ignore_index=True)
    climate_path = DATA_RAW / "Indian_Climate_Dataset_2024_2025.csv"
    climate.to_csv(climate_path, index=False)

    hours = pd.date_range("2023-01-01", "2025-10-31 23:00:00", freq="h")
    hourly_parts = []
    for city in CITIES:
        region = region_for(city)
        lat, lon = _COORDS[city]
        months = hours.month.values
        hod = hours.hour.values
        aqi_d = (
            _BASE_AQI[city]
            + _season_boost(months, region)
            + _diwali_boost(hours)
            + 12 * np.sin(hod / 24 * 2 * np.pi)
            + rng.normal(0, 20, len(hours))
        )
        aqi_d = np.clip(aqi_d, 15, 490)
        pm25 = np.clip(aqi_d * 0.55 + rng.normal(0, 8, len(hours)), 5, 450)
        pm10 = np.clip(pm25 * 1.35 + rng.normal(0, 10, len(hours)), 8, 550)
        rain = np.where(np.isin(months, [6, 7, 8, 9]) & (rng.random(len(hours)) < 0.15), rng.exponential(1.2, len(hours)), 0.0)
        temp = 20 + 9 * np.sin((months - 3) / 12 * 2 * np.pi) + 4 * np.sin(hod / 24 * 2 * np.pi) + rng.normal(0, 1.5, len(hours))
        hourly_parts.append(
            pd.DataFrame(
                {
                    "City": city,
                    "State": region,
                    "Latitude": lat,
                    "Longitude": lon,
                    "Datetime": hours,
                    "Season": pd.Series(months).map(SEASON_MONTHS).values,
                    "Festival_Period": ((hours.month == 10) & (hours.day >= 20) & (hours.day <= 31)).astype(int),
                    "Crop_Burning_Season": ((hours.month.isin([10, 11])) & (region == "IGP")).astype(int),
                    "Temp_2m_C": np.round(temp, 2),
                    "Humidity_Percent": np.clip(60 + rng.normal(0, 10, len(hours)), 15, 99),
                    "Wind_Speed_10m_kmh": np.clip(rng.normal(9, 3.5, len(hours)), 0.5, 40),
                    "Wind_Dir_10m": rng.uniform(0, 360, len(hours)),
                    "Precipitation_mm": np.round(rain, 2),
                    "Rain_mm": np.round(rain, 2),
                    "Pressure_MSL_hPa": np.round(rng.normal(1008, 3, len(hours)), 1),
                    "Cloud_Cover_Percent": np.clip(rng.normal(42, 18, len(hours)), 0, 100),
                    "PM2_5_ugm3": np.round(pm25, 2),
                    "PM10_ugm3": np.round(pm10, 2),
                    "CO_ugm3": np.round(np.clip(rng.normal(800, 120, len(hours)), 100, 3000), 1),
                    "NO2_ugm3": np.round(np.clip(rng.normal(35, 10, len(hours)), 2, 200), 2),
                    "SO2_ugm3": np.round(np.clip(rng.normal(12, 4, len(hours)), 1, 80), 2),
                    "O3_ugm3": np.round(np.clip(40 + 25 * (hod > 11).astype(float) + rng.normal(0, 8, len(hours)), 5, 220), 2),
                    "US_AQI": np.round(np.clip(aqi_d * 0.72, 5, 500), 1),
                    "EU_AQI": np.round(np.clip(aqi_d * 0.65, 5, 100), 1),
                    "AQI_Category": pd.cut(
                        aqi_d * 0.72,
                        bins=[-1, 50, 100, 150, 200, 300, 1000],
                        labels=["Good", "Moderate", "UnhealthySG", "Unhealthy", "VeryUnhealthy", "Hazardous"],
                    ).astype(str),
                    "Temp_Inversion": ((hod < 7) & (region == "IGP") & np.isin(months, [11, 12, 1])).astype(int),
                    "Wind_Stagnation": (rng.random(len(hours)) < 0.08).astype(int),
                    "Year": hours.year,
                    "Month": hours.month,
                    "Day": hours.day,
                }
            )
        )
    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly_path = DATA_RAW / "INDIA_AQI_COMPLETE_20251126.csv"
    hourly.to_csv(hourly_path, index=False)

    return {
        "cpcb": str(cpcb_path),
        "climate": str(climate_path),
        "hourly": str(hourly_path),
        "note": "synthetic placeholder — replace with Kaggle CSVs in data/raw/",
    }
