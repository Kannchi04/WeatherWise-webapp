import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# API Keys
RAPIDAPI_KEY = "ENTER_YOUR_RAPIDAPI_KEY"
OPENWEATHER_API_KEY = "ENTER_YOUR_OPENWEATHER_API_KEY"


HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
}


def get_lat_lon(city):
    geo_url = (
        f"https://api.openweathermap.org/geo/1.0/direct?"
        f"q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    )

    response = requests.get(geo_url)

    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return float(data["lat"]), float(data["lon"])

    return None, None


def get_station_id(lat, lon):
    url = "https://meteostat.p.rapidapi.com/stations/nearby"

    params = {
        "lat": lat,
        "lon": lon,
        "limit": 1
    }

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        result = response.json()

        if result.get("data"):
            return result["data"][0]["id"]

    return None


def get_historical_weather(station_id, start_date, end_date):
    url = "https://meteostat.p.rapidapi.com/stations/daily"

    params = {
        "station": station_id,
        "start": start_date,
        "end": end_date,
        "units": "metric"
    }

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        result = response.json()

        if result.get("data"):
            return pd.DataFrame(result["data"])

    return pd.DataFrame()


def get_start_date():
    return (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")


def get_end_date():
    return datetime.today().strftime("%Y-%m-%d")


def save_weather_data(df, city_name):
    filename = f"{city_name}_historical_weather.csv"
    df.to_csv(filename, index=False)
    return filename


def visualize_weather(df):
    plt.figure(figsize=(10, 5))

    plt.plot(df["date"], df["tavg"], marker='o', label="Avg Temp")
    plt.plot(df["date"], df["tmin"], linestyle='--', label="Min Temp")
    plt.plot(df["date"], df["tmax"], linestyle='--', label="Max Temp")

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Variation")

    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(8, 5))

    sns.histplot(df["wspd"], bins=15, kde=True)

    plt.xlabel("Wind Speed")
    plt.title("Wind Speed Distribution")

    plt.show()


def train_weather_model(df):
    df = df.dropna(axis=1, how='all')

    features = [
        col for col in [
            "tmin",
            "tmax",
            "prcp",
            "snow",
            "wspd",
            "wpgt",
            "pres"
        ]
        if col in df.columns
    ]

    df = df.dropna(subset=features + ["tavg"])

    X = df[features]
    y = df["tavg"]

    imputer = SimpleImputer(strategy="mean")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=features
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    performance = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    }

    return model, imputer, features, performance


def predict_future_temperature(model, imputer, features, future_data_dict):
    future_df = pd.DataFrame(future_data_dict)

    future_df = future_df[features]

    future_imputed = pd.DataFrame(
        imputer.transform(future_df),
        columns=features
    )

    prediction = model.predict(future_imputed)[0]

    return prediction
