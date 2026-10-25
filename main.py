import json
from pathlib import Path

import joblib
import pandas as pd

from src import charts
from src.features import WEATHER, add_features, to_daily
from src.forecast import AQI_FEATURES, error_by_city, horizon_table, model_rows, run
from src.load import load_climate, load_cpcb, load_hourly_as_daily

OUT = Path("outputs")
IMAGES = Path("images")


def main():
    OUT.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    cpcb = load_cpcb()
    weather = load_hourly_as_daily()
    climate = load_climate()

    # forecast table: CPCB cities that also have hourly weather, from the day weather starts
    cities = sorted(set(cpcb["city"]) & set(weather["city"]))
    start = weather["date"].min()
    aqi = cpcb[cpcb["city"].isin(cities) & (cpcb["date"] >= start)]
    daily = to_daily(aqi)
    daily = daily.merge(weather, on=["city", "date"], how="left")
    feats = add_features(daily)
    rows = model_rows(feats)

    _, _, res_aqi = run(rows, AQI_FEATURES)
    model, test, res = run(rows, AQI_FEATURES + WEATHER)
    by_city = error_by_city(test)
    horizons = horizon_table(feats, AQI_FEATURES + WEATHER)

    climate_check = climate.merge(cpcb, on=["city", "date"], how="inner")

    tables = {
        "cpcb_rows_raw": len(pd.read_csv("data/raw/AQI_2020_to_2025_October.csv", usecols=["City"])),
        "cpcb_city_days": len(cpcb),
        "cpcb_cities": int(cpcb["city"].nunique()),
        "cpcb_span": [str(cpcb["date"].min().date()), str(cpcb["date"].max().date())],
        "hourly_city_days": len(weather),
        "hourly_cities": int(weather["city"].nunique()),
        "hourly_span": [str(weather["date"].min().date()), str(weather["date"].max().date())],
        "climate_rows": len(climate),
        "climate_cities": int(climate["city"].nunique()),
        "climate_span": [str(climate["date"].min().date()), str(climate["date"].max().date())],
        "shared_cities": len(cities),
        "cpcb_days_in_forecast_window": int(daily["aqi"].notna().sum()),
        "calendar_days_after_reindex": len(daily),
        "with_weather": int(daily[["aqi", "temp"]].notna().all(axis=1).sum()),
        "model_rows": len(rows),
        "climate_matched_to_cpcb": len(climate_check),
    }

    findings = {
        "climate_vs_cpcb_corr": round(climate_check["climate_aqi"].corr(climate_check["aqi"]), 3),
        "us_vs_cpcb_corr": round(daily["us_aqi"].corr(daily["aqi"]), 3),
        "cities_beating_persistence": int((by_city["mae_model"] < by_city["mae_persistence"]).sum()),
        "cities_tested": len(by_city),
    }
    findings.update(charts.seasonal_findings(cpcb))

    results = {"tables": tables, "findings": findings, "forecast_aqi_only": res_aqi, "forecast_with_weather": res,
               "horizons": horizons.to_dict(orient="records")}
    print(json.dumps(results, indent=2, default=str))
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    by_city.round(2).to_csv(OUT / "error_by_city.csv")
    feats.to_parquet(OUT / "daily.parquet", index=False)
    joblib.dump({"model": model, "features": AQI_FEATURES + WEATHER, "cities": cities}, OUT / "model.joblib")

    charts.seasonal(cpcb, IMAGES / "seasonal_top_cities.png")
    charts.yearly(cpcb, IMAGES / "yearly_median.png")
    charts.forecast_city(test, "Delhi", IMAGES / "delhi_forecast.png")
    charts.city_error(by_city, IMAGES / "error_by_city.png")
    charts.scales(daily, climate_check, IMAGES / "aqi_scales.png")
    charts.horizons(horizons, IMAGES / "horizons.png")


if __name__ == "__main__":
    main()
