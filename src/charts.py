import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TOP_CITIES = ["Delhi", "Lucknow", "Kolkata", "Mumbai", "Bengaluru", "Chennai"]
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def full_years(cpcb):
    # 2025 stops in October, so year comparisons use 2020 to 2024
    return cpcb[cpcb["date"].dt.year <= 2024]


def seasonal_findings(cpcb):
    df = full_years(cpcb).copy()
    df["month"] = df["date"].dt.month
    delhi = df[df["city"] == "Delhi"].groupby("month")["aqi"].mean()
    by_month = df.groupby("month")["aqi"].median()
    days = df.groupby("city").size()
    big = days[days >= 1500].index
    worst = df[df["city"].isin(big)].groupby("city")["aqi"].mean().sort_values(ascending=False)
    yearly = df.groupby(df["date"].dt.year)["aqi"].median()
    delhi_bad = (df.loc[df["city"] == "Delhi", "aqi"] > 300).mean()
    return {
        "delhi_nov_mean": round(delhi[11], 1),
        "delhi_aug_mean": round(delhi[8], 1),
        "all_cities_worst_month": MONTHS[by_month.idxmax() - 1],
        "all_cities_best_month": MONTHS[by_month.idxmin() - 1],
        "worst_5_cities": worst.head(5).round(1).to_dict(),
        "yearly_median": yearly.round(1).to_dict(),
        "delhi_share_days_over_300": round(delhi_bad, 3),
    }


def seasonal(cpcb, path):
    df = full_years(cpcb)
    df = df[df["city"].isin(TOP_CITIES)]
    pivot = df.groupby([df["date"].dt.month, "city"])["aqi"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    for city in TOP_CITIES:
        ax.plot(range(1, 13), pivot[city], marker="o", label=city)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Mean CPCB AQI (2020 to 2024)")
    ax.set_title("Monthly AQI pattern in six large cities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def yearly(cpcb, path):
    df = full_years(cpcb)
    df = df[df["city"].isin(TOP_CITIES)]
    pivot = df.groupby([df["date"].dt.year, "city"])["aqi"].median().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for city in TOP_CITIES:
        ax.plot(pivot.index, pivot[city], marker="o", label=city)
    ax.set_xticks(pivot.index)
    ax.set_ylabel("Median CPCB AQI")
    ax.set_title("Median AQI by year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def forecast_city(test, city, path):
    sub = test[test["city"] == city].sort_values("date")
    tomorrow = sub["date"] + pd.Timedelta(days=1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(tomorrow, sub["target"], label="Actual", color="black", lw=1.2)
    ax.plot(tomorrow, sub["pred"], label="Model", lw=1)
    ax.plot(tomorrow, sub["persistence"], label="Same as today", lw=0.8, alpha=0.6)
    ax.set_ylabel("CPCB AQI")
    ax.set_title(f"Next-day AQI forecast for {city}, 2025 test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def city_error(by_city, path):
    df = by_city.sort_values("mae_model")
    fig, ax = plt.subplots(figsize=(8, 7))
    y = range(len(df))
    ax.barh([i + 0.2 for i in y], df["mae_model"], height=0.4, label="Model")
    ax.barh([i - 0.2 for i in y], df["mae_persistence"], height=0.4, label="Same as today")
    ax.set_yticks(list(y), df.index)
    ax.set_xlabel("Mean absolute error on 2025 test days (AQI points)")
    ax.set_title("Forecast error by city")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def scales(daily, climate_check, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    both = daily.dropna(subset=["aqi", "us_aqi"])
    axes[0].scatter(both["aqi"], both["us_aqi"], s=3, alpha=0.2)
    axes[0].set_xlabel("CPCB AQI")
    axes[0].set_ylabel("US AQI from hourly file (daily mean)")
    axes[0].set_title("Hourly file vs CPCB")
    axes[1].scatter(climate_check["aqi"], climate_check["climate_aqi"], s=3, alpha=0.3)
    axes[1].set_xlabel("CPCB AQI")
    axes[1].set_ylabel("AQI in climate file")
    axes[1].set_title("Climate file vs CPCB")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def horizons(table, path):
    df = table.set_index("horizon")[["mae_model", "mae_same_as_origin", "mae_mean_7d"]]
    df.columns = ["Model", "Same as origin day", "7-day mean at origin"]
    fig, ax = plt.subplots(figsize=(7, 4))
    df.plot.bar(ax=ax, rot=0)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("MAE on 2025 test days (AQI points)")
    ax.set_title("Forecast error by horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
