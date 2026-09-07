from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUT_FIGURES, OUT_REPORTS


def missingness(df: pd.DataFrame, name: str) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False).reset_index()
    miss.columns = ["column", "frac_missing"]
    miss.insert(0, "table", name)
    return miss


def write_profile(
    cpcb: pd.DataFrame,
    hourly: pd.DataFrame,
    climate: pd.DataFrame,
    fused: pd.DataFrame,
    extra: dict,
) -> None:
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    miss = pd.concat(
        [
            missingness(cpcb, "cpcb"),
            missingness(hourly, "hourly"),
            missingness(climate, "climate"),
            missingness(fused, "daily_fused"),
        ],
        ignore_index=True,
    )
    miss.to_csv(OUT_REPORTS / "missingness.csv", index=False)

    summary = {
        "cpcb_rows": int(len(cpcb)),
        "hourly_rows": int(len(hourly)),
        "climate_rows": int(len(climate)),
        "fused_rows": int(len(fused)),
        "fused_cities": sorted(fused["city"].unique().tolist()) if len(fused) else [],
        **extra,
    }
    (OUT_REPORTS / "phase1_profile.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _plot_city_timeseries(fused)
    _plot_aqi_scales(fused)
    _plot_seasonal(cpcb)


def _plot_city_timeseries(fused: pd.DataFrame) -> None:
    if fused.empty or "cpcb_aqi" not in fused.columns:
        return
    cities = list(fused.groupby("city")["cpcb_aqi"].size().sort_values(ascending=False).head(4).index)
    fig, axes = plt.subplots(len(cities), 1, figsize=(11, 2.4 * max(len(cities), 1)), sharex=True)
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities):
        sub = fused.loc[fused["city"] == city].sort_values("date")
        ax.plot(sub["date"], sub["cpcb_aqi"], lw=0.8, color="#1f4e79", label="CPCB AQI")
        if "us_aqi_mean" in sub.columns:
            ax.plot(sub["date"], sub["us_aqi_mean"], lw=0.7, color="#c45911", alpha=0.85, label="US AQI (daily mean)")
        ax.set_ylabel("AQI")
        ax.set_title(city)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "phase1_aqi_timeseries.png", dpi=140)
    plt.close(fig)


def _plot_aqi_scales(fused: pd.DataFrame) -> None:
    if fused.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "us_aqi_mean" in fused.columns:
        axes[0].scatter(fused["cpcb_aqi"], fused["us_aqi_mean"], s=4, alpha=0.15, c="#1f4e79")
        axes[0].set_xlabel("CPCB AQI")
        axes[0].set_ylabel("US AQI (daily mean)")
        axes[0].set_title("Do not mix these scales")
    if "climate_aqi" in fused.columns:
        both = fused.dropna(subset=["cpcb_aqi", "climate_aqi"])
        axes[1].scatter(both["cpcb_aqi"], both["climate_aqi"], s=4, alpha=0.2, c="#2e7d32")
        axes[1].set_xlabel("CPCB AQI")
        axes[1].set_ylabel("Climate-table AQI")
        axes[1].set_title("2024–25 overlap check")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "phase1_aqi_scale_scatter.png", dpi=140)
    plt.close(fig)


def _plot_seasonal(cpcb: pd.DataFrame) -> None:
    if cpcb.empty:
        return
    tmp = cpcb.copy()
    tmp["month"] = tmp["date"].dt.month
    pivot = tmp.pivot_table(index="month", columns="city", values="cpcb_aqi", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    for city in pivot.columns:
        ax.plot(pivot.index, pivot[city], marker="o", lw=1.2, label=city)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean CPCB AQI")
    ax.set_title("Seasonal AQI by city (CPCB bulletins)")
    ax.set_xticks(range(1, 13))
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "phase1_seasonal_aqi.png", dpi=140)
    plt.close(fig)
