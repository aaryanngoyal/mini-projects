from typing import List, Optional
from pydantic import BaseModel, Field

class DailyWeather(BaseModel):
    """Defines format for weather"""

    date: str = Field(description="YYYY-MM-DD")
    condition: str
    warning: Optional[str] = Field(default=None, description="Alert if bad weather") 

class HotelOption(BaseModel):
    """Defines format for hotel accomodation"""

    name: str
    price: float
    currency: str = "USD"
    review_score: Optional[float] = None

class TravelPlan(BaseModel):
    """Defines format for travel plan"""

    description: str
    checkin: str
    checkout: str
    weather_summary: str = Field(description="One line summary of weather suring stay")
    daily_weather: List[DailyWeather]
    hotels: List[HotelOption]
    recommendation: str = Field(description="Final personalized advice")