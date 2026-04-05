# LangGraph Travel Assistant Agent with Web Search + Phoenix tracing and observability

A simple LangGraph agent that can search the web using DuckDuckGo as well as leverage travel API tools, exposed via a FastAPI server, with plugged in Phoenix tracing and observability.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- OpenAI API key
- TicketMaster API key
- FourSquare API Key
- AviationStack API Key
- [Phoenix CLI](https://arize.com/docs/phoenix/self-hosting/deployment-options/terminal)

## Setup

1. Install dependencies:

```bash
poetry install
```
```bash
pip install arize-phoenix
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

3. Add your API keys to the `.env` file:

```
OPENAI_API_KEY=your_actual_api_key
FOURSQUARE_API_KEY=your_actual_api_key
TICKETMASTER_API_KEY=your_actual_api_key
AVIATIONSTACK_API_KEY=your_actual_api_key
```

## Running the API

Start the FastAPI server:

```bash
poetry run uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

Start Phoenix:

```bash
phoenix serve
```

Phoenix will be available at `http://localhost:6006`.

## Agent API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What concerts are happening in Denver in May?"}'
```

**Response:**

```json
{
  "response": "Based on my search, here are concerts in Denver..."
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

## Run user frustration and tool error evals

```bash
python phoenix_local_eval.py
```

## Project Structure

```
se-interview/
├── pyproject.toml            # Poetry dependencies
├── .env.example              # Environment variable template
├── README.md                 # This file
├── phoenix_local_eval.py     # LLM-as-a-judge implementation + tool error tracking  
├── agent.py                  # LangGraph agent implementation
├── api.py                    # FastAPI server
└── requirements.txt          # required packages          
```

## How It Works

1. The agent receives a user message via the `/chat` endpoint
2. It calls GPT-4o with the message and available tools (DuckDuckGo search, travel API tools)
3. If the LLM decides to search, it executes the search and feeds results back
4. The loop continues until the LLM provides a final response
5. The response is returned to the user
