import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

TEST_START = "2025-01-01"
HORIZONS = [1, 3, 7]
AQI_FEATURES = ["aqi", "aqi_yesterday", "aqi_week_ago", "aqi_mean_7d", "month", "dow", "city_code"]
PARAM_GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 15},
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
]


def model_rows(df):
    # only days where today, the 7-day history and tomorrow were all reported
    need = ["aqi", "aqi_yesterday", "aqi_week_ago", "aqi_mean_7d", "target"]
    rows = df.dropna(subset=need).copy()
    rows["city_code"] = rows["city"].astype("category").cat.codes
    return rows.sort_values("date").reset_index(drop=True)


def split(rows):
    test = rows["date"] >= pd.Timestamp(TEST_START)
    return rows[~test].copy(), rows[test].copy()


def make_model(params):
    return HistGradientBoostingRegressor(max_iter=300, random_state=0, **params)


def fill_with_train_medians(train, test, features):
    medians = train[features].median()
    return train[features].fillna(medians), test[features].fillna(medians)


def pick_params(X, y):
    # X and y are sorted by date, so each fold trains on the past only
    scores = []
    for params in PARAM_GRID:
        maes = []
        for fit_idx, val_idx in TimeSeriesSplit(n_splits=4).split(X):
            m = make_model(params).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            maes.append(mean_absolute_error(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
        scores.append(np.mean(maes))
    best = int(np.argmin(scores))
    return PARAM_GRID[best], scores


def run(rows, features):
    train, test = split(rows)
    X_train, X_test = fill_with_train_medians(train, test, features)
    params, cv_scores = pick_params(X_train, train["target"])
    model = make_model(params).fit(X_train, train["target"])
    test["pred"] = model.predict(X_test)
    test["persistence"] = test["aqi"]
    test["mean_7d"] = test["aqi_mean_7d"]
    result = {
        "n_train": len(train),
        "n_test": len(test),
        "params": params,
        "cv_mae": round(min(cv_scores), 2),
        "mae_model": round(mean_absolute_error(test["target"], test["pred"]), 2),
        "mae_persistence": round(mean_absolute_error(test["target"], test["persistence"]), 2),
        "mae_mean_7d": round(mean_absolute_error(test["target"], test["mean_7d"]), 2),
    }
    return model, test, result


def error_by_city(test):
    rows = []
    for city, g in test.groupby("city"):
        rows.append({
            "city": city,
            "days": len(g),
            "mae_model": mean_absolute_error(g["target"], g["pred"]),
            "mae_persistence": mean_absolute_error(g["target"], g["persistence"]),
        })
    return pd.DataFrame(rows).set_index("city").sort_values("mae_model")


def horizon_rows(feats):
    # Add tomorrow, 3 days and 7 days ahead as targets. Features stay as of the origin day.
    # Only origins where all three targets exist are kept, so every horizon is scored on the same days.
    df = feats.copy()
    for h in HORIZONS:
        df[f"target_{h}"] = df.groupby("city")["aqi"].shift(-h)
    rows = model_rows(df)
    return rows.dropna(subset=[f"target_{h}" for h in HORIZONS]).reset_index(drop=True)


def horizon_table(feats, features):
    rows = horizon_rows(feats)
    out = []
    for h in HORIZONS:
        rows["target"] = rows[f"target_{h}"]
        _, _, res = run(rows, features)
        best = min(res["mae_persistence"], res["mae_mean_7d"])
        out.append({
            "horizon": h,
            "test_days": res["n_test"],
            "mae_model": res["mae_model"],
            "mae_same_as_origin": res["mae_persistence"],
            "mae_mean_7d": res["mae_mean_7d"],
            "gain_vs_best_baseline_pct": round(100 * (best - res["mae_model"]) / best, 1),
        })
    return pd.DataFrame(out)
