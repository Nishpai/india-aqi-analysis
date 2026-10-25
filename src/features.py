import pandas as pd

WEATHER = ["temp", "humidity", "wind", "rain", "pressure", "pm25"]


def to_daily(df):
    # one row per city per calendar day, missing days become NaN rows
    parts = []
    for city, g in df.groupby("city"):
        g = g.set_index("date").sort_index()
        days = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(days)
        g.index.name = "date"
        g["city"] = city
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def add_features(daily):
    # everything here is known by the end of day t; the target is day t+1
    df = daily.sort_values(["city", "date"]).copy()
    by_city = df.groupby("city")["aqi"]
    df["aqi_yesterday"] = by_city.shift(1)
    df["aqi_week_ago"] = by_city.shift(7)
    df["aqi_mean_7d"] = by_city.transform(lambda s: s.rolling(7, min_periods=7).mean())
    df["target"] = by_city.shift(-1)
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.day_of_week
    return df
