import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# API Keys (Replace with actual keys)
RAPIDAPI_KEY = "f4c3b6b3f4msh7c27dcfe9cbeb30p18287djsn44ae080c26de"
OPENWEATHER_API_KEY = "8f1f0fc87531ec5f381cb58eca0c83f8"

# Headers for Meteostat API
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
}

def get_lat_lon(city):
    """Fetch latitude and longitude for a city using OpenWeather API."""
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(geo_url)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

def get_station_id(lat, lon):
    """Find nearest Meteostat weather station."""
    url = "https://meteostat.p.rapidapi.com/stations/nearby"
    params = {"lat": lat, "lon": lon, "limit": 1}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200 and response.json()["data"]:
        return response.json()["data"][0]["id"]
    return None

def get_historical_weather(station_id, start_date, end_date):
    """Fetch historical weather data from Meteostat API."""
    url = "https://meteostat.p.rapidapi.com/stations/daily"
    params = {"station": station_id, "start": start_date, "end": end_date, "units": "metric"}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200 and response.json()["data"]:
        return pd.DataFrame(response.json()["data"])
    return pd.DataFrame()

def get_start_date():
    """Get start date for historical weather data."""
    return (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

def get_end_date():
    """Get end date for historical weather data."""
    return datetime.today().strftime("%Y-%m-%d")

def save_weather_data(df, city_name):
    """Save weather data to a CSV file."""
    filename = f"{city_name}_historical_weather.csv"
    df.to_csv(filename, index=False)
    return filename

def visualize_weather(df):
    """Plot temperature variations and wind speed distribution."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["tavg"], marker='o', linestyle='-', label="Avg Temp")
    plt.plot(df["date"], df["tmin"], linestyle='--', label="Min Temp")
    plt.plot(df["date"], df["tmax"], linestyle='--', label="Max Temp")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Variation Over Time")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df["wspd"], bins=15, kde=True, color="purple")
    plt.xlabel("Wind Speed (m/s)")
    plt.title("Wind Speed Distribution")
    plt.show()

def train_weather_model(df):
    df = df.dropna(axis=1, how='all')

    features = [col for col in ["tmin", "tmax", "prcp", "snow", "wspd", "wpgt", "pres"] if col in df.columns]
    df = df.dropna(subset=features + ["tavg", "tmin", "tmax"])  # drop rows with NaNs in any target

    X = df[features]
    
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    X_train, X_test, y_train_avg, y_test_avg = train_test_split(X_imputed, df["tavg"], test_size=0.2, random_state=42)
    _, _, y_train_min, y_test_min = train_test_split(X_imputed, df["tmin"], test_size=0.2, random_state=42)
    _, _, y_train_max, y_test_max = train_test_split(X_imputed, df["tmax"], test_size=0.2, random_state=42)

    model_avg = LinearRegression()
    model_min = LinearRegression()
    model_max = LinearRegression()

    model_avg.fit(X_train, y_train_avg)
    model_min.fit(X_train, y_train_min)
    model_max.fit(X_train, y_train_max)

    return {
        "avg": model_avg,
        "min": model_min,
        "max": model_max
    }, imputer, features


def predict_future_temperature(models_dict, imputer, features, future_data_dict):
    future_data = {col: future_data_dict[col] for col in features if col in future_data_dict}
    future_df = pd.DataFrame(future_data)
    future_data_imputed = pd.DataFrame(imputer.transform(future_df), columns=features)

    avg_pred = models_dict["avg"].predict(future_data_imputed)[0]
    min_pred = models_dict["min"].predict(future_data_imputed)[0]
    max_pred = models_dict["max"].predict(future_data_imputed)[0]

    return {
        "avg": avg_pred,
        "min": min_pred,
        "max": max_pred
    }


def main():
    city_name = input("Enter city name: ")
    lat, lon = get_lat_lon(city_name)
    
    if lat and lon:
        station_id = get_station_id(lat, lon)
        if station_id:
            start_date = get_start_date()
            end_date = get_end_date()
            df = get_historical_weather(station_id, start_date, end_date)
            
            if not df.empty:
                filename = save_weather_data(df, city_name)
                print(f"✅ Data saved to {filename}")
                visualize_weather(df)
                
                model, imputer, features, performance = train_weather_model(df)
                print(f"Model Performance: {performance}")
                
                future_data_dict = {"tmin": [15], "tmax": [28], "prcp": [0], "snow": [0], "wspd": [7], "wpgt": [10], "pres": [1015]}
                predicted_temp = predict_future_temperature(model, imputer, features, future_data_dict)
                print(f"🌡️ Predicted Average Temperature: {predicted_temp:.2f}°C")
            else:
                print("❌ No weather data available.")
        else:
            print("❌ Could not find a weather station.")
    else:
        print("❌ Invalid city name or location data unavailable.")

if __name__ == "__main__":
    main()
