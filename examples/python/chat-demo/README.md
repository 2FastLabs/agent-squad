# Chat Demo

A Streamlit-based chat interface demonstrating the Agent Squad's intelligent routing capabilities with multiple specialized agents.

![Chat Demo](../img/chat-demo-screenshot.png)

## Overview

This demo showcases the core functionality of the Agent Squad framework - automatically routing user queries to the most appropriate specialized agent based on the content of the message.

## Available Agents

| Agent | Description | Capabilities |
|-------|-------------|--------------|
| **Tech Agent** | Technology expert | Software development, programming, AI, cloud computing, cybersecurity |
| **Health Agent** | Wellness advisor | General health, nutrition, fitness, mental health (no medical diagnoses) |
| **Weather Agent** | Weather assistant | Current weather conditions for any city worldwide |
| **Math Agent** | Math helper | Calculations, mathematical concepts, problem solving |

## How It Works

1. User enters a message in the chat interface
2. The **BedrockClassifier** analyzes the message content
3. The classifier routes the message to the most appropriate agent
4. The selected agent processes the request and responds
5. The UI displays which agent handled the request

## Prerequisites

- Python 3.8 or higher
- AWS account with access to Amazon Bedrock
- AWS credentials configured ([How to configure AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html))
- Claude models enabled in Amazon Bedrock ([Enable Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html))

## Running the Demo

From the `examples/python` directory:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main app (includes this demo)
streamlit run main-app.py
```

Or run this demo directly:

```bash
cd examples/python/chat-demo
streamlit run chat-demo.py
```

## Example Queries

Try these example queries to see the routing in action:

- **Tech**: "What is Docker and how does containerization work?"
- **Health**: "What are some tips for better sleep hygiene?"
- **Weather**: "What's the weather like in Tokyo?"
- **Math**: "Calculate 15% of 250"

## Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│ BedrockClassifier│
│ (Claude Haiku)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Agent Selection             │
├──────────┬──────────┬───────────────┤
│          │          │               │
▼          ▼          ▼               ▼
Tech     Health    Weather          Math
Agent    Agent     Agent           Agent
│          │          │               │
└──────────┴──────────┴───────────────┘
                   │
                   ▼
             Response
```

## Features

- **Persistent Chat History**: Conversation context maintained throughout the session
- **Agent Attribution**: See which agent handled each response
- **Streaming Responses**: Real-time response streaming for better UX
- **Tool Integration**: Weather and Math agents use external tools
- **Session Management**: Clear chat history and start fresh sessions

## Customization

You can easily extend this demo by:

1. Adding new specialized agents
2. Implementing additional tools
3. Customizing agent system prompts
4. Adjusting the classifier model

See the [Agent Squad documentation](https://awslabs.github.io/agent-squad/) for more details on customization options.
