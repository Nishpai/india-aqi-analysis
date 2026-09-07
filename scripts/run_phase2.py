from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.clusters import cluster_cities, seasonal_fingerprints
from src.config import DATA_PROCESSED, OUT_REPORTS
from src.forecast import fit_forecast_aqi
from src.pollutant import fit_prominent_pollutant


def main() -> None:
    cpcb_path = DATA_PROCESSED / "daily_cpcb_wide.parquet"
    fused_path = DATA_PROCESSED / "daily_fused.parquet"
    if not cpcb_path.exists() or not fused_path.exists():
        raise SystemExit("Run scripts/run_phase1.py first.")

    cpcb = pd.read_parquet(cpcb_path)
    fused = pd.read_parquet(fused_path)

    feat = seasonal_fingerprints(cpcb)
    clusters, cluster_meta = cluster_cities(feat)
    print("clusters k=", cluster_meta["k"], "silhouette=", cluster_meta["silhouette"])
    print(clusters[["city", "region", "cluster", "cluster_label", "aqi_mean"]].to_string(index=False))

    pol = fit_prominent_pollutant(fused)
    print("prominent pollutant macro-F1", pol["macro_f1"], "vs majority", pol["majority_baseline_macro_f1"])

    fc = fit_forecast_aqi(fused)
    print("forecast MAE model", fc["model"]["mae"], "persistence", fc["persistence"]["mae"])

    summary = {
        "clusters": cluster_meta,
        "prominent_pollutant_macro_f1": pol["macro_f1"],
        "forecast": fc["model"],
        "forecast_persistence": fc["persistence"],
    }
    (OUT_REPORTS / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
