# Downloads the three Kaggle datasets into data/raw.
# Needs a Kaggle API token set up for the kaggle command line tool.
import subprocess

DATASETS = [
    "krishnandansha/aqi-india-2020-to-2025-october",
    "bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025",
    "ankushnarwade/indian-climate-dataset-20242025",
]

for slug in DATASETS:
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", "data/raw", "--unzip"], check=True)
