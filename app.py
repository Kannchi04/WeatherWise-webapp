import streamlit as st
import requests
import API  # Import API key and function from api.py
import model  # Import model-related functions
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🌦 Weatherwise - Your Personal Weather Website")

city = st.text_input("Enter city name:", "Indore")

if st.button("Get Weather"):
    # Fetch current weather
    data = API.get_weather(city)

    if data.get("cod") != 200:
        st.error(f"Error: {data.get('message')}")
    else:
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        st.success(f"**Weather in {city}**")
        st.write(f"🌡 Temperature: {temp}°C")
        st.write(f"💧 Humidity: {humidity}%")
        st.write(f"💨 Wind Speed: {wind_speed} m/s")
        st.write(f"🌤 Condition: {weather_desc.capitalize()}")

        # Fetch historical weather data
        lat, lon = model.get_lat_lon(city)
        if lat and lon:
            station_id = model.get_station_id(lat, lon)
            if station_id:
                df = model.get_historical_weather(station_id, model.get_start_date(), model.get_end_date())
                
                if not df.empty:
                    # Visualize weather trends
                    st.write("### 📊 Temperature Trends")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df["date"], df["tavg"], marker='o', linestyle='-', label="Avg Temp", color="blue")
                    ax.plot(df["date"], df["tmin"], linestyle='--', label="Min Temp", color="green")
                    ax.plot(df["date"], df["tmax"], linestyle='--', label="Max Temp", color="red")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title("Temperature Variation Over Time")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)

                    # Wind Speed Distribution
                    st.write("### 🌪️ Wind Speed Distribution")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    sns.histplot(df["wspd"], bins=15, kde=True, color="purple", ax=ax2)
                    ax2.set_xlabel("Wind Speed (m/s)")
                    ax2.set_title("Wind Speed Distribution")
                    st.pyplot(fig2)

                    # Predict future temperature
                    models_dict, imputer, features = model.train_weather_model(df)
                    future_data_dict = {"tmin": [15], "tmax": [28], "prcp": [0], "snow": [0], "wspd": [7], "wpgt": [10], "pres": [1015]}

                    prediction = model.predict_future_temperature(models_dict, imputer, features, future_data_dict)

                    st.write(f"🌡️ Predicted Avg Temp: {prediction['avg']:.2f}°C")
                    st.write(f"🌤️ Predicted Min Temp: {prediction['min']:.2f}°C")
                    st.write(f"🔥 Predicted Max Temp: {prediction['max']:.2f}°C")

                else:
                    st.warning("⚠️ No historical weather data available.")
            else:
                st.error("❌ Could not find a weather station.")
        else:
            st.error("❌ Invalid city name or location data unavailable.")