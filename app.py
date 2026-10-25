# Demo of the next-day AQI forecast. Run main.py first, then: streamlit run app.py

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# CPCB AQI bands: upper limit, name, colour
BANDS = [
    (50, "Good", "green"),
    (100, "Satisfactory", "green"),
    (200, "Moderate", "orange"),
    (300, "Poor", "orange"),
    (400, "Very Poor", "red"),
    (500, "Severe", "red"),
]


def category(aqi):
    for limit, name, colour in BANDS:
        if aqi <= limit:
            return name, colour
    return "Severe", "red"


st.set_page_config(page_title="India AQI forecast", layout="wide")
st.title("India city AQI: trend and next-day forecast")
st.write("Daily CPCB AQI for 25 cities and a next-day forecast from the last reported day.")

if not Path("outputs/model.joblib").exists():
    st.error("Model not found. Run python main.py first.")
    st.stop()


@st.cache_data
def load_data():
    daily = pd.read_parquet("outputs/daily.parquet")
    results = json.loads(Path("outputs/results.json").read_text())
    by_city = pd.read_csv("outputs/error_by_city.csv", index_col=0)
    return daily, results, by_city


@st.cache_resource
def load_model():
    return joblib.load("outputs/model.joblib")


daily, results, by_city = load_data()
saved = load_model()
model, features, cities = saved["model"], saved["features"], saved["cities"]

city = st.sidebar.selectbox("City", cities, index=cities.index("Delhi"))
days = st.sidebar.slider("Days to show", 30, 730, 180)

sub = daily[daily["city"] == city].sort_values("date")
last = sub.dropna(subset=["aqi", "aqi_yesterday", "aqi_week_ago", "aqi_mean_7d"]).tail(1).copy()
last["city_code"] = cities.index(city)
pred = float(model.predict(last[features])[0])
day = last["date"].iloc[0]
next_day = day + pd.Timedelta(days=1)
name, colour = category(pred)

overall = results["forecast_with_weather"]
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"AQI on {day.date()}", f"{last['aqi'].iloc[0]:.0f}")
col2.metric(f"Forecast for {next_day.date()}", f"{pred:.0f}")
col3.metric(f"Model MAE, {city} (2025)", f"{by_city.loc[city, 'mae_model']:.1f}")
col4.metric(f"Same as today MAE, {city}", f"{by_city.loc[city, 'mae_persistence']:.1f}")
st.markdown(f"Forecast category: :{colour}[**{name}**]")
st.caption(f"All 25 cities: model MAE {overall['mae_model']}, same as today {overall['mae_persistence']}, "
           f"7-day mean {overall['mae_mean_7d']}.")

recent = sub.tail(days)
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(recent["date"], recent["aqi"], label="CPCB AQI")
ax.plot([next_day], [pred], "o", color="red", markersize=8, label="forecast")
ax.set_ylabel("AQI")
ax.legend(loc="upper left")
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

st.subheader("Last 10 days")
table = sub.tail(10)[["date", "aqi", "aqi_mean_7d", "temp", "humidity", "wind", "rain"]].copy()
table["date"] = table["date"].dt.date
st.dataframe(table.round(1), hide_index=True, width="stretch")

with st.expander("How it works"):
    st.write(
        "The model is a HistGradientBoostingRegressor trained on 2022-08 to 2024-12. It uses today's AQI, "
        "yesterday's, a week ago, the 7-day mean, month, day of week, city and today's weather from the "
        "hourly file. Days next to a gap in the data are not used. It was tested on 2025 against two "
        "baselines: tomorrow is the same as today, and tomorrow is the 7-day mean."
    )
