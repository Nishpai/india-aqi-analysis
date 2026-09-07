from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_REPORTS = ROOT / "outputs" / "reports"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_MODELS = ROOT / "outputs" / "models"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUT_REPORTS, OUT_FIGURES, OUT_MODELS):
    _p.mkdir(parents=True, exist_ok=True)

CLIMATE_GLOBS = ("*climate*", "*Climate*")
HOURLY_GLOBS = (
    "INDIA_AQI*.csv",
    "*indian_air_quality*",
    "*air_quality*2022*",
    "*COMPLETE*.csv",
)
CPCB_GLOBS = ("AQI_2020*", "*2020_to_2025*", "*bulletin*")

# Walk-forward split used by Phase 2 supervised models.
TRAIN_END = "2024-12-31"
VALID_END = "2025-06-30"

INDIAN_AQI_BREAKPOINTS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]

SEASON_MONTHS = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-Monsoon",
    11: "Post-Monsoon",
}


def indian_aqi_category(value: float) -> str:
    if value != value:
        return "Unknown"
    for cap, label in INDIAN_AQI_BREAKPOINTS:
        if value <= cap:
            return label
    return "Severe"
