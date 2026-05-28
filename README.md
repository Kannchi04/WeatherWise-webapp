# WeatherWise - Weather Forecasting & Analytics Web App

## Project Overview

WeatherWise is a weather forecasting and analytics web application that provides real-time weather information, weather visualizations, and simple predictive insights using weather APIs and machine learning.

The project integrates OpenWeather API and Meteostat API to fetch current and historical weather data, visualize weather trends, and predict future temperature using Linear Regression.

---

## Key Files

* API.py : Handles real-time weather API requests
* app.html : Frontend weather dashboard interface
* model.py : Historical weather analysis and machine learning model

---

## Features

* Real-time weather information
* City-based weather search
* Temperature, humidity, and wind insights
* Weather data visualizations using Chart.js
* Historical weather data analysis
* Machine learning temperature prediction
* Interactive frontend dashboard

---

## Technologies Used

* Python
* HTML
* CSS
* JavaScript
* Chart.js
* OpenWeather API
* Meteostat API
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Functionalities

### Real-Time Weather Dashboard

The web application provides:

* Current temperature
* Humidity
* Wind speed
* Weather condition description

### Weather Visualization

The dashboard includes:

* Temperature trend charts
* Wind speed visualization
* Interactive weather graphs

### Historical Weather Analysis

The model fetches:

* Historical temperature data
* Wind speed data
* Pressure and precipitation details

### Machine Learning Prediction

A Linear Regression model is trained using historical weather data to:

* Predict future average temperature
* Analyze weather trends

---

## Machine Learning Workflow

The model.py script performs:

* Data collection from APIs
* Data preprocessing
* Missing value handling
* Feature selection
* Model training using Linear Regression
* Model evaluation using:

  * MAE
  * RMSE
  * R² Score

---

## APIs Used

### OpenWeather API

Used for:

* Current weather data
* Geolocation coordinates

### Meteostat API

Used for:

* Historical weather records
* Station-based weather analytics

---

## How to Run

### 1. Clone Repository

```bash id="hqp1oh"
git clone https://github.com/your-username/WeatherWise.git
cd WeatherWise
```

### 2. Install Dependencies

```bash id="h3g0np"
pip install -r requirements.txt
```

### 3. Add API Keys

Replace placeholders in the code:

```python id="ry4u1j"
API_KEY = "YOUR_API_KEY"
```

and

```python id="5g3s5m"
RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY"
```

---

### 4. Run Backend Scripts

```bash id="k2z2kn"
python API.py
```

or

```bash id="tbjlwm"
python model.py
```

---

### 5. Open Frontend

Open `app.html` in your browser.

---

## Demo Preview

<img width="706" height="882" alt="image" src="https://github.com/user-attachments/assets/4be21c51-1755-4e92-95e2-9726b9561f3c" />

---

## Future Improvements

* 7-day weather forecasting
* Live weather maps
* Weather alerts and notifications
* Deep learning forecasting models
* Flask/Django backend integration
* Deployment on cloud platforms

---

## Learning Outcomes

This project helped in understanding:

* API integration
* Weather data analytics
* Frontend visualization
* Machine learning workflow
* Data preprocessing
* Predictive modeling

---

## Requirements

* Python 3.x
* Internet connection
* Valid API keys for:

  * OpenWeather API
  * RapidAPI Meteostat API
