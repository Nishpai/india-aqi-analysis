from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import OUT_FIGURES, OUT_REPORTS, SEASON_MONTHS
from src.city_map import region_for


def seasonal_fingerprints(cpcb: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    df = cpcb.dropna(subset=["cpcb_aqi", "city", "date"]).copy()
    df["season"] = df["date"].dt.month.map(SEASON_MONTHS)
    counts = df.groupby("city").size()
    keep = counts[counts >= min_days].index
    df = df[df["city"].isin(keep)]
    seasonal = df.pivot_table(index="city", columns="season", values="cpcb_aqi", aggfunc="mean")
    extra = df.groupby("city")["cpcb_aqi"].agg(aqi_mean="mean", aqi_std="std", aqi_p90=lambda s: s.quantile(0.9))
    winter = seasonal.get("Winter")
    monsoon = seasonal.get("Monsoon")
    extra["winter_minus_monsoon"] = (winter - monsoon) if winter is not None and monsoon is not None else np.nan
    extra["region"] = extra.index.map(region_for)
    feat = seasonal.join(extra)
    return feat.reset_index()


def cluster_cities(feat: pd.DataFrame, k_min: int = 3, k_max: int = 6) -> tuple[pd.DataFrame, dict]:
    num_cols = [c for c in feat.columns if c not in {"city", "region"}]
    X = feat[num_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n = len(feat)
    k_max = min(k_max, max(2, n - 1))
    k_min = min(k_min, k_max)
    scores = []
    best_k, best_score, best_labels = k_min, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=7)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(Xs, labels))
        scores.append({"k": k, "silhouette": sil})
        if sil > best_score:
            best_k, best_score, best_labels = k, sil, labels
    if best_labels is None:
        km = KMeans(n_clusters=2, n_init=20, random_state=7)
        best_labels = km.fit_predict(Xs)
        best_k = 2
        best_score = float("nan")

    out = feat.copy()
    out["cluster"] = best_labels
    # Readable names from winter AQI of cluster centroid.
    winter_col = "Winter" if "Winter" in out.columns else "aqi_mean"
    names = {}
    for c in sorted(out["cluster"].unique()):
        mu = out.loc[out["cluster"] == c, winter_col].mean()
        names[c] = f"cluster_{c}_winterAQI_{mu:.0f}"
    out["cluster_label"] = out["cluster"].map(names)

    pca = PCA(n_components=2, random_state=7)
    xy = pca.fit_transform(Xs)
    out["pca1"] = xy[:, 0]
    out["pca2"] = xy[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, sub in out.groupby("cluster"):
        ax.scatter(sub["pca1"], sub["pca2"], s=40, label=names[c])
        for _, row in sub.iterrows():
            ax.annotate(row["city"], (row["pca1"], row["pca2"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("City clusters from seasonal CPCB AQI fingerprints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "city_clusters_pca.png", dpi=140)
    plt.close(fig)

    out.to_csv(OUT_REPORTS / "city_clusters.csv", index=False)
    meta = {"k": int(best_k), "silhouette": best_score, "scores": scores, "pc_var": pca.explained_variance_ratio_.tolist()}
    (OUT_REPORTS / "city_clusters_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta
