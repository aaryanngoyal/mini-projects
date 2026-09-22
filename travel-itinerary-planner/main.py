import os
import json

from google import genai
from dotenv import load_dotenv

from tools import Tools
from schema import TravelPlan

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

tool_map = {
    "get_current_temperature" : Tools.get_current_temperature,
    "get_hotels": Tools.get_hotels,
}

all_tools = [Tools.weather_tool, Tools.hotel_tool]

system_prompt = """
You are a travel planning assistant. When a user asks for hotel recommendations:

1. FIRST, call `get_weather_forecast` for the location and travel dates to check for 
   rain, storms, or snow. If bad weather is forecast, warn the user clearly.
2. THEN, call `get_hotels` with the location, check-in, check-out, and budget 
   to find available hotels.
3. FINALLY, summarize: mention weather warnings first, then list hotels.

Always extract location, dates, and budget from the user's message. 
If any are missing, ask the user before calling tools.
"""

def run_agent(user_query: str, max_iter: int = 5) -> str:
    """Agent for getting information about hotel availability and weather"""


    interaction = client.interactions.create(
        model="gemini-3.1-flash-lite",
        input=user_query,
        tools=all_tools,
        system_instruction=system_prompt,
    )

    for iter in range(max_iter):
        function_calls = [s for s in interaction.steps if s.type == "function_call"]

        if not function_calls:
            return interaction.output_text

        results = []
        for call in function_calls:
            fn = tool_map.get(call.name)
            if not fn:
                results.append({
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": json.dumps({"error": "unknown tool"})}],
                })
                continue

            output = fn(**call.arguments)
            results.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": json.dumps(output)}],
            })

        interaction = client.interactions.create(
            model="gemini-3.1-flash-lite",
            input=results,
            tools=all_tools,
            system_instruction=system_prompt,
            previous_interaction_id=interaction.id
        )

    if iter == max_iter-1 and function_calls:
        interaction = client.interactions.create(
            model="gemini-3.1-flash-lite",
            input="Please provide final answer now based on the information gathered.",
            tools=[],
            previous_interaction_id=interaction.id
        )   
        return interaction.output_text

    final_response = client.interactions.create(
        model="gemini-3.1-flash-lite",
        input="Based on all the tools above, produce a complete travel plan.",
        previous_interaction_id=interaction.id,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": TravelPlan.model_json_schema(),
        },
    )

    plan = TravelPlan.model_validate_json(final_response.output_text)
    return plan.recommendation

if __name__ == "__main__":

    user_query = "Find me a hotel in Delhi from 2026-09-23 to 2026-09-26 under $200"
    print(run_agent(user_query))