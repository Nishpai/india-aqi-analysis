import pandas as pd

from src.features import add_features, to_daily
from src.forecast import horizon_rows, model_rows, split
from src.load import clean_city, load_cpcb


def make_aqi(dates, city="Delhi"):
    dates = pd.to_datetime(dates)
    return pd.DataFrame({"city": city, "date": dates, "aqi": range(100, 100 + len(dates))})


def test_clean_city():
    assert clean_city("Greater_Noida") == "Greater Noida"
    assert clean_city("Aurangabad\n(Maharashtra)") == "Aurangabad (Maharashtra)"
    assert clean_city(" Delhi ") == "Delhi"


def test_cpcb_dates_are_day_first(tmp_path):
    path = tmp_path / "cpcb.csv"
    path.write_text("Date,City,AQI Value\n13-01-2020,Delhi,300\n13-01-2020,Delhi,310\n")
    df = load_cpcb(path)
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-13")
    assert len(df) == 1 and df["aqi"].iloc[0] == 305


def test_gap_days_are_not_used():
    dates = list(pd.date_range("2024-01-01", "2024-01-10")) + list(pd.date_range("2024-01-13", "2024-01-25"))
    feats = add_features(to_daily(make_aqi(dates)))
    rows = model_rows(feats)
    # target is always the next calendar day's value
    nxt = feats.set_index("date")["aqi"]
    for _, r in rows.iterrows():
        assert r["target"] == nxt[r["date"] + pd.Timedelta(days=1)]
    # days next to the gap have no target or no full history
    assert pd.Timestamp("2024-01-10") not in set(rows["date"])
    assert pd.Timestamp("2024-01-13") not in set(rows["date"])


def test_features_do_not_see_the_future():
    dates = pd.date_range("2024-01-01", "2024-02-29")
    base = add_features(to_daily(make_aqi(dates)))
    changed = make_aqi(dates)
    changed.loc[changed["date"] > "2024-02-01", "aqi"] = 999
    after = add_features(to_daily(changed))
    cols = ["aqi_yesterday", "aqi_week_ago", "aqi_mean_7d"]
    day = base["date"] <= "2024-02-01"
    pd.testing.assert_frame_equal(base.loc[day, cols], after.loc[day, cols])


def test_split_is_by_date():
    dates = pd.date_range("2024-12-01", "2025-01-31")
    rows = model_rows(add_features(to_daily(make_aqi(dates))))
    train, test = split(rows)
    assert train["date"].max() < test["date"].min()
    assert test["date"].min() == pd.Timestamp("2025-01-01")


def test_horizon_targets_are_h_days_after_the_origin():
    dates = pd.date_range("2024-01-01", "2024-02-29")
    feats = add_features(to_daily(make_aqi(dates)))
    rows = horizon_rows(feats)
    aqi = feats.set_index("date")["aqi"]
    for _, r in rows.iterrows():
        assert r["aqi"] == aqi[r["date"]]  # features are from the origin day
        for h in (1, 3, 7):
            assert r[f"target_{h}"] == aqi[r["date"] + pd.Timedelta(days=h)]
    # the last 7 origins have no 7-day target and are dropped
    assert rows["date"].max() == pd.Timestamp("2024-02-22")
