# India AQI analysis

Joint pipeline for three Kaggle sources: CPCB daily bulletins (gold Indian AQI), hourly city atmosphere (features), and the 2024–2025 climate table (overlap check).

## Setup

```
python -m pip install -r requirements.txt
```

Put CSVs in `data/raw/`, or download with a Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`:

```
python scripts/download_kaggle.py
```

If those files are missing, Phase 1 writes **synthetic** CSVs with the same column names so the code path still runs. Replace them with the real extracts before trusting metrics.

## Run

```
python scripts/run_all.py
```

Or `python scripts/run_phase1.py` then `python scripts/run_phase2.py`.

Phase 1 writes `data/processed/*.parquet` and `outputs/reports/phase1_profile.json`.  
Phase 2 writes city clusters, prominent-pollutant metrics, and next-day CPCB AQI forecast vs persistence.

Targets: next-day CPCB AQI (not US AQI); prominent pollutant from weather/calendar only (no same-day species).
