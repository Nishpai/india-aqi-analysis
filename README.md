# India AQI analysis

An analysis of daily CPCB (Central Pollution Control Board) air quality bulletins for 295 Indian cities from 2020 to 2025, plus a next-day AQI forecast for 25 cities checked against simple baselines.

## Results at a glance

- **Winter is the problem season.** Across all cities the worst month is November and the best is July. Delhi's mean AQI is 354.5 in November against 90.4 in August, and it was above 300 ("very poor" or worse) on 21.5% of days from 2020 to 2024.
- **Next-day forecast:** MAE **16.72** AQI points on 5,894 test days in 2025, against **17.34** for "same as today" and 20.87 for the 7-day mean. That is about a 3.6% gain, and the model beats "same as today" in 17 of 25 cities.
- **Data issues found:** a date-parsing trap that silently dropped or swapped most dates, a "prominent pollutant" column that is the same on every row, and a third-party AQI column that does not match CPCB at all (correlation -0.02).

![Monthly pattern](images/seasonal_top_cities.png)

Stack: pandas, scikit-learn (`HistGradientBoostingRegressor`, `TimeSeriesSplit`), matplotlib, Streamlit, pytest.

## Problem and data

Air quality in Indian cities swings a lot by season and place. The questions: when and where is it worst, and can tomorrow's AQI be predicted better than "same as today"?

Three Kaggle datasets go in `data/raw/` (not in this repo):

| Source | Span | Size |
|---|---|---|
| [CPCB daily AQI bulletins](https://www.kaggle.com/datasets/krishnandansha/aqi-india-2020-to-2025-october) | 2020-01-01 to 2025-10-31 | 378,328 rows, 295 cities after name cleanup |
| [Hourly weather and pollution](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025) | 2022-08-05 to 2025-11-26 | 842,160 hourly rows, 29 cities, 35,090 city-days |
| [Indian climate 2024 to 2025](https://www.kaggle.com/datasets/ankushnarwade/indian-climate-dataset-20242025) | 2024-01-01 to 2025-12-31 | 7,310 rows, 10 cities |

Rows kept at each step:

| Step | Rows |
|---|---|
| CPCB raw rows | 378,328 |
| CPCB city-days after fixing city spellings and averaging 34 duplicate rows | 378,294 |
| CPCB city-days for the 25 cities in both CPCB and the hourly file, from 2022-08-05 | 27,170 |
| Same cities reindexed to every calendar day (missing days left empty) | 28,885 |
| CPCB days that also have weather | 27,170 |
| Days usable for the forecast (today, the last 7 days and tomorrow all reported) | 23,689 |
| Climate file rows that match a CPCB city-day | 6,689 |

## How it works

1. `src/load.py` reads the three CSVs, parses CPCB dates as day-first, merges city names spelled two ways, and averages duplicate city-days.
2. `src/features.py` reindexes each city to every calendar day, so "yesterday" and "tomorrow" are really one day away, and adds today's AQI, yesterday's, a week ago and the 7-day mean.
3. Today's weather (temperature, humidity, wind, rain, pressure, PM2.5) is added from the hourly file. Days next to a gap are dropped instead of being filled.
4. `src/forecast.py` splits by date (train 2022-08 to 2024-12, test 2025), fills missing weather with training medians, and picks the learning rate and leaf count with a 4-fold `TimeSeriesSplit` on training data only.
5. The model is compared with two baselines: tomorrow equals today, and tomorrow equals the 7-day mean. The same setup is repeated for 3 and 7 days ahead.
6. `main.py` runs everything and writes `outputs/results.json` and the charts in `images/`. Every number in this README comes from that file.

## Findings

Year and month comparisons use the full years 2020 to 2024.

1. **Winter is the problem season.** The worst month by median AQI is November and the best is July. In Delhi, mean AQI is 354.5 in November and 90.4 in August, about 4 times higher.
2. **Delhi was above 300 on 21.5% of days.**
3. **The worst cities are all around Delhi.** Among cities with at least 1,500 reported days: Greater Noida (206.1), Delhi (203.2), Ghaziabad (198.8), Bhiwadi (194.1) and Faridabad (193.8).
4. **No clear yearly trend.** Median AQI across all cities: 88 (2020), 93 (2021), 97 (2022), 94 (2023), 88 (2024).
5. **The other two sources do not agree with CPCB.** The hourly file's US AQI correlates 0.47 with CPCB AQI (it is a different scale). The climate file's AQI correlates -0.02 with CPCB for the same city and day, so it is not used for anything.

![Yearly median](images/yearly_median.png)
![AQI scales](images/aqi_scales.png)

## Forecast results

Mean absolute error on the 5,894 test days (2025-01-09 to 2025-10-30):

| Method | MAE (AQI points) |
|---|---|
| 7-day mean | 20.87 |
| Same as today | 17.34 |
| Model, AQI history only | 16.91 |
| Model, AQI history and weather | 16.72 |

The model beats "same as today" in 17 of 25 cities, but only by about 3.6% overall, and weather adds only a little.

![Delhi forecast](images/delhi_forecast.png)
![Error by city](images/error_by_city.png)

**Longer horizons.** One model per horizon, a forecast made at the end of day t for day t+h using only what is known on day t. Only origin days with a reported value 1, 3 and 7 days later are used, which leaves 5,574 test days, so the 1-day numbers differ slightly from the table above.

| Days ahead | Model MAE | Same as origin day | 7-day mean at origin | Gain vs best baseline |
|---|---|---|---|---|
| 1 | 16.57 | 17.31 | 20.69 | 4.3% |
| 3 | 21.88 | 25.61 | 23.50 | 6.9% |
| 7 | 24.32 | 28.90 | 26.16 | 7.0% |

The gain grows a little, from 4.3% to about 7%, but the model's own error still rises from 16.6 to 24.3 AQI points. At 3 and 7 days the 7-day mean is the stronger baseline.

![Horizons](images/horizons.png)

## What I found in the data

- **Date format.** CPCB dates are day-first (`13-01-2020`). Read month-first, most dates are silently dropped or swapped, and the data appears to run past its real end date.
- **City spellings.** Some cities are spelled two ways in different years (Greater_Noida and Greater Noida, Kalaburgi and Kalaburagi, and others). These are merged in `src/load.py`.
- **A useless label.** The "Prominent Pollutant" column is `PM 2.5, PM 10` on every single row, so there is nothing to predict and it is not used.
- **Third-party AQI.** The climate file's AQI has no relation to CPCB (correlation -0.02), so its source is unclear.

## Limitations

- The gain over "same as today" is small (about 3.6% at 1 day), measured on one test period (2025).
- Weather is the same day's observed weather, not a weather forecast, so this is a "given today's conditions" model.
- Some cities repeat the same AQI for days in a row (Imphal has only 16 distinct values across its 137 reported days in 2025), which makes "same as today" hard to beat there.
- The hourly file covers only 29 cities from August 2022, so the forecast covers 25 cities, not all 295.

## Demo

![Streamlit app](images/app.png)

`streamlit run app.py` lets you pick a city, see its recent AQI, and get a forecast with its CPCB category for the day after its last reported day.

## How to run

Needs Python 3.12+ and a Kaggle API token. Create one under Kaggle Settings, API, and save it as `~/.kaggle/access_token` (on Windows `%USERPROFILE%\.kaggle\access_token`). The hourly CSV is about 280 MB once unzipped.

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python main.py
python -m pytest
streamlit run app.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python download_data.py
python main.py
python -m pytest
streamlit run app.py
```

`python -m pytest` runs 6 tests: city cleanup, day-first dates, gap days not used, no future values in features, the date split, and horizon targets being exactly h days after the origin.

## Project structure

```text
download_data.py         downloads the three Kaggle datasets into data/raw
main.py                  runs the pipeline and writes outputs/ and images/
src/load.py              reads and cleans the three CSVs
src/features.py          daily reindex and lag features
src/forecast.py          split, tuning, model, baselines and horizons
src/charts.py            findings and charts
app.py                   Streamlit demo
tests/test_pipeline.py   tests for cleanup, dates, gaps, leakage, split and horizons
requirements.txt         pinned Python packages
images/                  charts and the app screenshot used in this README
```
