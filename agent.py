import os
import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool

import requests

# Implement Phoenix Tracing for better observability
from phoenix.otel import register

# Register tracing
tracer_provider = register(
    project_name="travel-assistant-agent",
    auto_instrument=True
)

load_dotenv()

# -----------------------------
# API Keys
# -----------------------------

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")

# -----------------------------
# Tools
# -----------------------------

@tool
def find_flights(origin: str, destination: str, date: str) -> dict:
    """Find flights between two airports on a specific date using IATA codes."""
    
    url = "http://api.aviationstack.com/v1/flights"

    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "dep_iata": origin,
        "arr_iata": destination,
        "flight_date": date,
        "limit": 5
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        flights = resp.json().get("data", [])

        return {
            "origin": origin,
            "destination": destination,
            "flights": [
                {
                    "airline": f.get("airline", {}).get("name"),
                    "flight_number": f.get("flight", {}).get("iata"),
                    "departure_airport": f.get("departure", {}).get("airport"),
                    "arrival_airport": f.get("arrival", {}).get("airport")
                }
                for f in flights[:5]
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def find_places(location: str, query: str) -> dict:
    """Search for activities, restaurants, or points of interest in a city."""

    url = "https://api.foursquare.com/v3/places/search"

    headers = {
        "Authorization": FOURSQUARE_API_KEY
    }

    params = {
        "near": location,
        "query": query,
        "limit": 5
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()

        results = resp.json().get("results", [])

        return {
            "location": location,
            "results": [
                {
                    "name": r.get("name"),
                    "address": r.get("location", {}).get("formatted_address"),
                    "category": r.get("categories", [{}])[0].get("name")
                }
                for r in results
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def find_events(location: str, keyword: str) -> dict:
    """Find events such as concerts, sports, or shows in a given city."""

    url = "https://app.ticketmaster.com/discovery/v2/events.json"

    params = {
        "apikey": TICKETMASTER_API_KEY,
        "city": location,
        "keyword": keyword,
        "size": 5
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        events = resp.json().get("_embedded", {}).get("events", [])

        return {
            "location": location,
            "events": [
                {
                    "name": e.get("name"),
                    "date": e.get("dates", {}).get("start", {}).get("localDate"),
                    "venue": e.get("_embedded", {}).get("venues", [{}])[0].get("name")
                }
                for e in events
            ]
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Tool Registration
# -----------------------------

tools = [
    find_flights,
    find_places,
    find_events
]

tools_by_name = {tool.name: tool for tool in tools}

# -----------------------------
# Model Setup
# -----------------------------

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

# -----------------------------
# Graph State
# -----------------------------

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# -----------------------------
# LLM Node
# -----------------------------

def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""
                            You are a travel assistant that helps users find flights, places to visit, restaurants, and events.

                            You have access to the following tools:

                            find_flights:
                            Use this when a user asks about flights between airports.
                            Inputs:
                            - origin: origin airport IATA code (e.g., PDX)
                            - destination: destination airport IATA code (e.g., LAX)
                            - date: flight date in YYYY-MM-DD format

                            find_places:
                            Use this when a user asks about restaurants, attractions, or things to do.
                            Inputs:
                            - location: city name
                            - query: type of place (e.g., restaurant, museum, park)

                            find_events:
                            Use this when a user asks about concerts, shows, or events.
                            Inputs:
                            - location: city name
                            - keyword: event type (concert, comedy, sports, etc.)

                            Always prefer calling a tool if it can answer the question.
                            Return helpful summaries of the tool results.
                            """
)
                ]
                + state["messages"]
            )
        ]
    }

# -----------------------------
# Tool Node
# -----------------------------

def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""

    result = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

        observation = tool.invoke(tool_call["args"])

        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": result}

# -----------------------------
# Routing Logic
# -----------------------------

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END

# -----------------------------
# Agent Builder
# -----------------------------

def build_agent():

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")

    graph_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )

    graph_builder.add_edge("tool_node", "llm_call")

    agent = graph_builder.compile()

    return agent

# debug API issues
print("Foursquare key loaded:", FOURSQUARE_API_KEY[:5])