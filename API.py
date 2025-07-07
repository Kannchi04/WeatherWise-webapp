import requests

API_KEY = "8f1f0fc87531ec5f381cb58eca0c83f8"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)
    return response.json()

city = "Indore"
data = get_weather(city)
print(data)
