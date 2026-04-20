"""Weather tool for the chat demo using Open-Meteo API."""

import urllib.request
import urllib.parse
import json
from agent_squad.utils import Logger


def get_weather(city: str) -> str:
    """
    Get current weather conditions for a city using Open-Meteo API.

    :param city: The city name to get weather for
    """
    try:
        Logger.info(f"Getting weather for: {city}")

        # First, geocode the city name to get coordinates
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"

        with urllib.request.urlopen(geocode_url, timeout=10) as response:
            geocode_data = json.loads(response.read().decode())

        if not geocode_data.get('results'):
            return f"Could not find location: {city}"

        location = geocode_data['results'][0]
        lat = location['latitude']
        lon = location['longitude']
        location_name = location.get('name', city)
        country = location.get('country', '')

        # Get current weather using coordinates
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            f"&temperature_unit=celsius"
        )

        with urllib.request.urlopen(weather_url, timeout=10) as response:
            weather_data = json.loads(response.read().decode())

        current = weather_data.get('current', {})

        # Weather code descriptions
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail",
        }

        weather_code = current.get('weather_code', 0)
        weather_desc = weather_codes.get(weather_code, "Unknown")

        result = (
            f"Weather in {location_name}, {country}:\n"
            f"- Conditions: {weather_desc}\n"
            f"- Temperature: {current.get('temperature_2m', 'N/A')}C\n"
            f"- Humidity: {current.get('relative_humidity_2m', 'N/A')}%\n"
            f"- Wind Speed: {current.get('wind_speed_10m', 'N/A')} km/h"
        )

        return result

    except urllib.error.URLError as e:
        Logger.error(f"Network error getting weather for {city}: {e}")
        return f"Network error getting weather for {city}. Please try again."
    except Exception as e:
        Logger.error(f"Error getting weather for {city}: {e}")
        return f"Error getting weather for {city}: {str(e)}"
