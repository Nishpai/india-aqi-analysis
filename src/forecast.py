from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import OUT_FIGURES, OUT_MODELS, OUT_REPORTS
from src.features import add_calendar, add_lags, time_split


def _gb_regressor():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=7,
            verbosity=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=400, random_state=7)


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_forecast_aqi(fused: pd.DataFrame) -> dict:
    df = fused.dropna(subset=["cpcb_aqi", "date", "city"]).copy()
    df = add_calendar(df)
    lag_cols = [c for c in ("cpcb_aqi", "pm25_mean", "us_aqi_mean") if c in df.columns]
    df = add_lags(df, "city", lag_cols, lags=[1, 7])
    df["target"] = df.groupby("city")["cpcb_aqi"].shift(-1)
    df["target_date"] = df.groupby("city")["date"].shift(-1)
    df = df.dropna(subset=["target", "cpcb_aqi_lag1"])
    df = time_split(df)

    weather = [
        c
        for c in (
            "temp_mean",
            "humidity_mean",
            "wind_speed_mean",
            "rain_sum",
            "pressure_mean",
            "festival",
            "crop_burning",
            "temp_inversion",
            "wind_stagnation",
            "month",
            "dow",
            "dayofyear",
        )
        if c in df.columns
    ]
    lags = [c for c in df.columns if "_lag" in c or "_roll7" in c]
    df["city_code"] = pd.Categorical(df["city"]).codes
    features = weather + lags + ["city_code"]
    features = list(dict.fromkeys(features))

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df["target"]
    train = df["split"] == "train"
    test = df["split"] == "test"
    if test.sum() < 80:
        test = df["split"].isin(["valid", "test"])

    persist = df.loc[test, "cpcb_aqi"]  # today -> tomorrow
    model = _gb_regressor()
    model.fit(X.loc[train], y.loc[train])
    pred = model.predict(X.loc[test])

    payload = {
        "model": _metrics(y.loc[test], pred),
        "persistence": _metrics(y.loc[test], persist),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "features": features,
        "target": "next-day CPCB AQI",
        "split": "walk-forward by date (train through 2024 when available)",
    }
    # Per-city MAE on test
    tmp = df.loc[test, ["city", "target"]].copy()
    tmp["pred"] = pred
    tmp["persist"] = persist.to_numpy()
    per_city = (
        tmp.groupby("city", group_keys=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mae_model": mean_absolute_error(g["target"], g["pred"]),
                    "mae_persist": mean_absolute_error(g["target"], g["persist"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_city.to_csv(OUT_REPORTS / "forecast_mae_by_city.csv", index=False)
    payload["beats_persistence_cities"] = int((per_city["mae_model"] < per_city["mae_persist"]).sum())

    joblib.dump({"model": model, "features": features}, OUT_MODELS / "forecast_aqi.joblib")
    (OUT_REPORTS / "forecast_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _plot_forecast_city(df.loc[test], pred, persist.to_numpy(), city="Delhi")
    return payload


def _plot_forecast_city(test_df: pd.DataFrame, pred: np.ndarray, persist: np.ndarray, city: str) -> None:
    sub_idx = test_df["city"] == city
    if sub_idx.sum() < 10:
        city = test_df["city"].iloc[0]
        sub_idx = test_df["city"] == city
    fig, ax = plt.subplots(figsize=(11, 4))
    dates = test_df.loc[sub_idx, "target_date"]
    ax.plot(dates, test_df.loc[sub_idx, "target"], lw=1.2, label="Actual next-day CPCB AQI", color="#1f4e79")
    ax.plot(dates, pred[sub_idx.to_numpy()], lw=1.0, label="Model", color="#c45911")
    ax.plot(dates, persist[sub_idx.to_numpy()], lw=0.8, alpha=0.7, label="Persistence", color="#7f7f7f")
    ax.set_title(f"Next-day AQI forecast — {city} (test window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "forecast_test.png", dpi=140)
    plt.close(fig)
