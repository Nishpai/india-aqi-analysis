from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import OUT_MODELS, OUT_REPORTS
from src.features import add_calendar, primary_pollutant, time_split


def _gb_classifier():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=7,
            verbosity=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300, random_state=7)


def _season_code(s: pd.Series) -> pd.Series:
    order = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post-Monsoon": 3}
    return s.map(order).fillna(-1)


def fit_prominent_pollutant(fused: pd.DataFrame) -> dict:
    df = fused.dropna(subset=["prominent_pollutant", "date", "city"]).copy()
    df["y"] = df["prominent_pollutant"].map(primary_pollutant)
    df = df[df["y"] != "Unknown"]
    df = add_calendar(df)
    df = time_split(df)

    feature_cols = []
    for col in (
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
    ):
        if col in df.columns:
            feature_cols.append(col)
    df["season_code"] = _season_code(df["season"])
    feature_cols.append("season_code")
    df["region_code"] = LabelEncoder().fit_transform(df["region"].astype(str)) if "region" in df.columns else 0
    if "region" in df.columns:
        feature_cols.append("region_code")
    df["city_code"] = LabelEncoder().fit_transform(df["city"].astype(str))
    feature_cols.append("city_code")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df["y"]
    mask = y.groupby(y).transform("size") >= 30
    df, X, y = df.loc[mask], X.loc[mask], y.loc[mask]

    train = df["split"] == "train"
    test = df["split"] == "test"
    if test.sum() < 50:
        test = df["split"].isin(["valid", "test"])

    model = _gb_classifier()
    model.fit(X.loc[train], y.loc[train])
    pred = model.predict(X.loc[test])
    report = classification_report(y.loc[test], pred, output_dict=True, zero_division=0)
    macro = float(f1_score(y.loc[test], pred, average="macro", zero_division=0))

    # Majority-class baseline
    maj = y.loc[train].mode().iloc[0]
    base = float(f1_score(y.loc[test], np.full(test.sum(), maj), average="macro", zero_division=0))

    joblib.dump({"model": model, "features": feature_cols}, OUT_MODELS / "prominent_pollutant.joblib")
    payload = {
        "macro_f1": macro,
        "majority_baseline_macro_f1": base,
        "majority_class": maj,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "report": report,
        "features": feature_cols,
        "note": "Weather + calendar + city only; same-day PM/O3 not used (would leak the label).",
    }
    (OUT_REPORTS / "prominent_pollutant_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
