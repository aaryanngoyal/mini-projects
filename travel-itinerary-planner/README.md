# Travel Itinerary Planner

An AI-powered travel planning assistant built with Google's Gemini API. Given a destination, travel dates, and a budget, it checks the weather forecast, searches for hotels, and produces a structured itinerary — all through Gemini's function-calling capabilities.

## Features

- **Weather-aware planning**: Fetches the multi-day forecast and warns about rain, storms, or snow during the trip.
- **Budget hotel search**: Finds available hotels within a specified price range using the Booking.com API (via RapidAPI).
- **Agentic tool use**: Gemini autonomously decides when and how to call each tool based on the user's query.
- **Structured output**: Returns the final plan as a validated Pydantic model (destination, weather summary, hotel options, and recommendation).
- **Resilient**: Includes error handling, retry logic, and a loop cap to prevent runaway tool calls.

## Architecture

```
User Query
   │
   ▼
┌──────────────────────────┐
│   Gemini Agent Loop      │  ← system prompt guides tool ordering
│  (max 5 iterations)      │
└──────────┬───────────────┘
           │
           ├──► get_weather_forecast(location, start_date, end_date)
           │       └── OpenWeather /forecast API
           │
           └──► get_hotels(location, checkin, checkout, budget)
                   ├── Booking.com searchDestination (resolve city → dest_id)
                   └── Booking.com searchHotels (with price_max filter)
           │
           ▼
┌──────────────────────────┐
│  Structured Output Phase │  ← response_json_schema (Pydantic)
└──────────────────────────┘
           │
           ▼
      TravelPlan JSON
```

## Prerequisites

- Python 3.10+
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/)
- An **OpenWeather API key** from [openweathermap.org](https://openweathermap.org/api)
- A **RapidAPI key** subscribed to the [Booking.com API](https://rapidapi.com/) (free Basic plan works)

## Installation

```bash
git clone <your-repo-url>
cd travel-itinerary-planner
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`

```
google-genai
python-dotenv
requests
pydantic
```

## Configuration

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
RAPID_API_KEY=your_rapidapi_key_here
```

**Never commit `.env` to version control.** Add it to `.gitignore`:

```
.env
venv/
__pycache__/
*.pyc
```

## Usage

```bash
python main.py
```

Then modify the `user_query` at the bottom of `main.py` to test different scenarios:

```python
user_query = "Plan a trip to Delhi from 2026-10-15 to 2026-10-18 with a budget of $200 for hotels."
print(run_agent(user_query))
```

### Example Output

```json
{
  "destination": "Delhi",
  "checkin": "2026-10-15",
  "checkout": "2026-10-18",
  "weather_summary": "Warm and mostly clear, with rain expected on Oct 16.",
  "daily_weather": [
    {"date": "2026-10-15", "condition": "clear", "warning": null},
    {"date": "2026-10-16", "condition": "rain", "warning": "Pack an umbrella."},
    {"date": "2026-10-17", "condition": "clouds", "warning": null}
  ],
  "hotels": [
    {"name": "Hotel A", "price": 150.0, "currency": "USD", "review_score": 8.2},
    {"name": "Hotel B", "price": 180.0, "currency": "USD", "review_score": 7.9}
  ],
  "recommendation": "Book Hotel A for the best value. Expect rain on the 16th, so plan indoor activities for that day."
}
```

## Project Structure

```
travel-itinerary-planner/
├── main.py           # Agent loop, tool dispatch, structured output
├── schema.py         # Defines pydantic model for structured output
├── tools.py          # Tools class: weather + hotel functions and schemas
├── .env              # API keys (not committed)
├── requirements.txt
└── README.md
```

## Key Implementation Details

### Tool Declarations (`tools.py`)

Each tool is defined as a JSON Schema that Gemini reads to understand when and how to call it:

```python
weather_tool = {
    "type": "function",
    "name": "get_weather_forecast",
    "description": "Gets the weather forecast for a location and date range.",
    "parameters": { ... },
}
```

### Agent Loop (`main.py`)

The loop executes function calls Gemini requests until it stops requesting them, capped at `max_iterations`:

```python
for iteration in range(max_iterations):
    function_calls = [s for s in interaction.steps if s.type == "function_call"]
    if not function_calls:
        break
    # execute each call, package result, send back
```

### Structured Output

The final phase uses `response_json_schema` with a Pydantic model. **Note**: Tools cannot be passed in the same request as `response_json_schema`, so this is a separate second-phase call.

## Constraints & Gotchas

| Constraint | Impact |
|---|---|
| **Gemini free tier RPD** | `gemini-3.x-flash` allows only 20 requests/day. Use `flash-lite` (500/day) for development. |
| **OpenWeather free forecast** | Only 5 days ahead. Farther dates return no data — handle gracefully. |
| **`price_max` is total stay price** | Not per-night. Clarify in the system prompt or schema description. |
| **Booking.com `dest_id`** | Must be resolved via `searchDestination` before calling `searchHotels`. |
| **Tool + structured output** | Cannot be combined in one API request. Use two phases. |

## License

MIT

## Acknowledgments

- [Google Gemini API](https://ai.google.dev/) for function calling and structured output
- [OpenWeather](https://openweathermap.org/) for forecast data
- [Booking.com API on RapidAPI](https://rapidapi.com/) for hotel search
