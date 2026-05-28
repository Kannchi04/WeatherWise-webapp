import requests

API_KEY = "ENTER_YOUR_API_KEY"
BASE_URL = "ENTER_YOUR_BASE_URL"


def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "cod": response.status_code,
            "message": response.json().get("message", "Error fetching weather data")
        }

#test
if __name__ == "__main__":
    city = "Indore"
    data = get_weather(city)
    print(data)
