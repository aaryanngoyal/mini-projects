import os
import requests
from dotenv import load_dotenv

load_dotenv()

class Tools:
    """
    Used to define tool parameters and their respective methods
    """

    # paramters for weather tool
    weather_tool = {
        "type": "function",
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location and date range. Use this to warn users about bad weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, eg. Delhi",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date YYYY-MM-DD",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date YYYY-MM-DD",
                },
            },
            "required": ["location", "start_date", "end_date"],
        }
    }

    # paramteres for hotel tool
    hotel_tool = {
        "type": "function",
        "name": "get_hotels",
        "decription": "Gets available hotels within budget for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, eg. Delhi",
                },
                "checkin": {
                    "type": "string",
                    "description": "Date of checkin in hotel, Format - YYYY-MM-DD",
                },
                "checkout": {
                    "type": "string",
                    "description": "Date of checkout from hotel, Format - YYYY-MM-DD",
                },
                "budget": {
                    "type": "integer",
                    "description": "Amount for hotel booking",
                },
            },
            "required": ["location", "checkin", "checkout", "budget"],
        }
    }

    @staticmethod
    def get_current_temperature(location: str, start_date: str, end_date: str) -> dict:
        """Gets current temperature of location"""

        API_KEY = os.getenv("OPENWEATHER_API_KEY")
        BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

        params = {
            "q": location,
            "appid": API_KEY,
            "units": "metric",
        }

        try:
            hotel_resp = requests.get(BASE_URL, params=params, timeout=10)
            hotel_resp.raise_for_status()
            data = hotel_resp.json()

            warnings = []
            for entry in data["list"]:
                entry_date = entry["dt_txt"].split(" ")[0]
                if start_date <= entry_date <= end_date:
                    weather = entry["weather"][0]["main"].lower()
                    if weather in ("rain", "thunderstorm", "snow"):
                        warnings.append({
                            "date": entry_date,
                            "warning": weather,
                            "temp": entry["main"]["temp"],
                        })

            return {
                "location": data["city"]["name"],
                "warnings": warnings,
                "summary": f"{len(warnings)} bad-weather days in range" if warnings else "Clear forecast",
            }

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    @staticmethod
    def get_hotels(location, checkin, checkout, budget):
        """Gets information about hotel availability for specified location within date range and budget"""

        BASE_URL = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels"
        DEST_URL = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination"

        headers = {
            "x-rapidapi-host": "booking-com15.p.rapidapi.com",
            "x-rapidapi-key": os.getenv("RAPID_API_KEY")
        }

        try:
            dest_response = requests.get(
                DEST_URL,
                headers=headers,
                params={
                    "query": location
                }
            )

            dest_response.raise_for_status()
            dest_data = dest_response.json().get("data", [])

            if not dest_data:
                return {"error": f"{location} not found"}

            dest_id = dest_data[0]["dest_id"]
            dest_type = dest_data[0]["dest_type"]

            querystring = {
                "dest_id": dest_id,
                "search_type": dest_type,
                "arrival_date": checkin,
                "departure_date": checkout,
                "adults": 2,
                "room_qty": 1,
                "currency_code": "USD",
                "price_min": "0",
                "price_max": str(budget),
                "page_number": "1",
                "languagecode": "en-us"
            }
            
            hotel_resp = requests.get(BASE_URL, headers=headers, params=querystring)
            hotel_resp.raise_for_status()
            data = hotel_resp.json()

            hotels = data.get("data", {}).get("hotel", [])

            results = []
            for h in hotels:
                prop = h.get("property", {})
                results.append({
                    "name": prop.get("name"),
                    "price": prop.get("priceBreakdown", {})
                                .get("grossPrice", {})
                                .get("value"),
                    "currency": "USD",
                    "review_score": prop.get("reviewScore"),
                    "review_count": prop.get("reviewCount"),
                })

            return {"hotels": results}

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}