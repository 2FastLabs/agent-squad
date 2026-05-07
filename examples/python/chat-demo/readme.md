# Chat Demo

Streamlit chat that routes queries to specialized agents.

## Agents

| Agent | Handles |
|-------|---------|
| Tech | Software, programming, AI, cloud |
| Health | Wellness, nutrition, fitness |
| Weather | Current conditions for any city |
| Math | Calculations, math concepts |

## How It Works

1. User sends a message
2. BedrockClassifier picks the right agent
3. Agent responds (with streaming)

## Example Queries

- "What's Docker and how does containerization work?"
- "Tips for better sleep?"
- "Weather in Tokyo?"
- "Calculate 15% of 250"

## Features

- Persistent chat history
- Shows which agent handled each response
- Real-time streaming
- Weather uses Open-Meteo API, Math uses safe expression evaluator

## Getting Started

See the [Python examples guide](../readme.md) for setup.

Then:
```bash
streamlit run chat-demo.py
```
