from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

CPCB_FILE = "AQI_2020_to_2025_October.csv"
HOURLY_FILE = "INDIA_AQI_COMPLETE_20251126.csv"
CLIMATE_FILE = "Indian_Climate_Dataset_2024_2025.csv"

# The same city is spelled two ways in different years of the CPCB file.
CITY_FIXES = {
    "Asanol": "Asansol",
    "GandhiNagar": "Gandhinagar",
    "Greater_Noida": "Greater Noida",
    "Kalaburgi": "Kalaburagi",
    "Manglore": "Mangalore",
    "Medikeri": "Madikeri",
    "Pimpri Chinchwad": "Pimpri-Chinchwad",
    "Tiruppur": "Tirupur",
    "Yamuna Nagar": "Yamunanagar",
}


def clean_city(name):
    name = " ".join(str(name).split())
    return CITY_FIXES.get(name, name)


def load_cpcb(path=None):
    df = pd.read_csv(path or RAW / CPCB_FILE)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], format="%d-%m-%Y"),
        "city": df["City"].map(clean_city),
        "aqi": df["AQI Value"].astype(float),
    })
    # a few city-days appear twice, keep their average
    return out.groupby(["city", "date"], as_index=False)["aqi"].mean()


def load_hourly_as_daily(path=None):
    cols = {
        "City": "city",
        "Datetime": "datetime",
        "Temp_2m_C": "temp",
        "Humidity_Percent": "humidity",
        "Wind_Speed_10m_kmh": "wind",
        "Rain_mm": "rain",
        "Pressure_MSL_hPa": "pressure",
        "PM2_5_ugm3": "pm25",
        "US_AQI": "us_aqi",
    }
    df = pd.read_csv(path or RAW / HOURLY_FILE, usecols=list(cols)).rename(columns=cols)
    df["city"] = df["city"].map(clean_city)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    daily = df.groupby(["city", "date"]).agg(
        temp=("temp", "mean"),
        humidity=("humidity", "mean"),
        wind=("wind", "mean"),
        rain=("rain", "sum"),
        pressure=("pressure", "mean"),
        pm25=("pm25", "mean"),
        us_aqi=("us_aqi", "mean"),
    )
    return daily.reset_index()


def load_climate(path=None):
    df = pd.read_csv(path or RAW / CLIMATE_FILE)
    return pd.DataFrame({
        "date": pd.to_datetime(df["Date"]),
        "city": df["City"].map(clean_city),
        "climate_aqi": df["AQI"].astype(float),
    })
