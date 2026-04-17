<h2 align="center">Agent Squad&nbsp;</h2>

<p align="center">Flexible and powerful framework for managing multiple AI agents and handling complex conversations.</p>


<p align="center">
  <a href="https://github.com/awslabs/agent-squad"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repo-green.svg" /></a>
  <a href="https://pypi.org/project/agent-squad/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agent-squad.svg?style=flat-square"></a>
  <a href="https://awslabs.github.io/agent-squad/"><img alt="Documentation" src="https://img.shields.io/badge/docs-book-blue.svg?style=flat-square"></a>
</p>

<p align="center">
  <!-- GitHub Stats -->
  <img src="https://img.shields.io/github/stars/awslabs/agent-squad?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/awslabs/agent-squad?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/watchers/awslabs/agent-squad?style=social" alt="GitHub watchers">
</p>

<p align="center">
  <!-- Repository Info -->
  <img src="https://img.shields.io/github/last-commit/awslabs/agent-squad" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/awslabs/agent-squad" alt="Issues">
  <img src="https://img.shields.io/github/issues-pr/awslabs/agent-squad" alt="Pull Requests">
</p>

<p align="center">
  <!-- Package Stats -->
  <a href="https://pypi.org/project/agent-squad/"><img src="https://img.shields.io/pypi/dm/agent-squad?label=pypi%20downloads" alt="PyPI Monthly Downloads"></a>
  <img src="https://img.shields.io/static/v1?label=python&message=%203.11|%203.12|%203.13&color=blue?style=flat-square&logo=python" alt="Python versions">

</p>

## 🔖 Features

- 🧠 **Intelligent intent classification** — Dynamically route queries to the most suitable agent based on context and content.
- 🌊 **Flexible agent responses** — Support for both streaming and non-streaming responses from different agents.
- 📚 **Context management** — Maintain and utilize conversation context across multiple agents for coherent interactions.
- 🔧 **Extensible architecture** — Easily integrate new agents or customize existing ones to fit your specific needs.
- 🌐 **Universal deployment** — Run anywhere - from AWS Lambda to your local environment or any cloud platform.
- 📦 **Pre-built agents and classifiers** — A variety of ready-to-use agents and multiple classifier implementations available.
- 🔤 **TypeScript support** — Native TypeScript implementation available.

## What's the Agent Squad ❓

The Agent Squad is a flexible framework for managing multiple AI agents and handling complex conversations. It intelligently routes queries and maintains context across interactions.

The system offers pre-built components for quick deployment, while also allowing easy integration of custom agents and conversation messages storage solutions.

This adaptability makes it suitable for a wide range of applications, from simple chatbots to sophisticated AI systems, accommodating diverse requirements and scaling efficiently.


## 🏗️ High-level architecture flow diagram

<br /><br />

![High-level architecture flow diagram](https://raw.githubusercontent.com/awslabs/agent-squad/main/img/flow.jpg)

<br /><br />


1. The process begins with user input, which is analyzed by a Classifier.
2. The Classifier leverages both Agents' Characteristics and Agents' Conversation history to select the most appropriate agent for the task.
3. Once an agent is selected, it processes the user input.
4. The orchestrator then saves the conversation, updating the Agents' Conversation history, before delivering the response back to the user.


## 💬 Demo App

To quickly get a feel for the Agent Squad, we've provided a Demo App with a few basic agents. This interactive demo showcases the orchestrator's capabilities in a user-friendly interface. To learn more about setting up and running the demo app, please refer to our [Demo App](https://awslabs.github.io/agent-squad/cookbook/examples/chat-demo-app/) section.

<br>

In the screen recording below, we demonstrate an extended version of the demo app that uses 6 specialized agents:
- **Travel Agent**: Powered by an Amazon Lex Bot
- **Weather Agent**: Utilizes a Bedrock LLM Agent with a tool to query the open-meteo API
- **Restaurant Agent**: Implemented as an Amazon Bedrock Agent
- **Math Agent**: Utilizes a Bedrock LLM Agent with two tools for executing mathematical operations
- **Tech Agent**: A Bedrock LLM Agent designed to answer questions on technical topics
- **Health Agent**: A Bedrock LLM Agent focused on addressing health-related queries

Watch as the system seamlessly switches context between diverse topics, from booking flights to checking weather, solving math problems, and providing health information.
Notice how the appropriate agent is selected for each query, maintaining coherence even with brief follow-up inputs.

The demo highlights the system's ability to handle complex, multi-turn conversations while preserving context and leveraging specialized agents across various domains.

Click on the image below to see a screen recording of the demo app on the GitHub repository of the project:
<a href="https://github.com/awslabs/agent-squad" target="_blank">
  <img src="https://raw.githubusercontent.com/awslabs/agent-squad/main/img/demo-app.jpg" alt="Demo App Screen Recording" style="max-width: 100%; height: auto;">
</a>



## 🚀 Getting Started

Check out our [documentation](https://awslabs.github.io/agent-squad/) for comprehensive guides on setting up and using the Agent Squad!



### Core Installation

```bash
# Optional: Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install agent-squad[aws]
```

#### Default Usage

Here's an equivalent Python example demonstrating the use of the Agent Squad with a Bedrock LLM Agent and a Lex Bot Agent:

```python
import sys
import asyncio
from agent_squad.orchestrator import AgentSquad
from agent_squad.agents import BedrockLLMAgent, LexBotAgent, BedrockLLMAgentOptions, LexBotAgentOptions, AgentStreamResponse

orchestrator = AgentSquad()

tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
  name="Tech Agent",
  streaming=True,
  description="Specializes in technology areas including software development, hardware, AI, \
  cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs \
  related to technology products and services.",
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
))
orchestrator.add_agent(tech_agent)


health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
  name="Health Agent",
  streaming=True,
  description="Specializes in health and well being",
))
orchestrator.add_agent(health_agent)

async def main():
    # Example usage
    response = await orchestrator.route_request(
        "What is AWS Lambda?",
        'user123',
        'session456',
        {},
        True
    )

    # Handle the response (streaming or non-streaming)
    if response.streaming:
        print("\n** RESPONSE STREAMING ** \n")
        # Send metadata immediately
        print(f"> Agent ID: {response.metadata.agent_id}")
        print(f"> Agent Name: {response.metadata.agent_name}")
        print(f"> User Input: {response.metadata.user_input}")
        print(f"> User ID: {response.metadata.user_id}")
        print(f"> Session ID: {response.metadata.session_id}")
        print(f"> Additional Parameters: {response.metadata.additional_params}")
        print("\n> Response: ")

        # Stream the content
        async for chunk in response.output:
            if isinstance(chunk, AgentStreamResponse):
                print(chunk.text, end='', flush=True)
            else:
                print(f"Received unexpected chunk type: {type(chunk)}", file=sys.stderr)
        print()

    else:
        # Handle non-streaming response (ConversationMessage)
        print("\n** RESPONSE ** \n")
        print(f"> Agent ID: {response.metadata.agent_id}")
        print(f"> Agent Name: {response.metadata.agent_name}")
        print(f"> User Input: {response.metadata.user_input}")
        print(f"> User ID: {response.metadata.user_id}")
        print(f"> Session ID: {response.metadata.session_id}")
        print(f"> Additional Parameters: {response.metadata.additional_params}")
        print(f"\n> Response: {response.output.content}")

if __name__ == "__main__":
  asyncio.run(main())
```

The following example demonstrates how to use the Agent Squad with two different types of agents: a Bedrock LLM Agent with Converse API support and a Lex Bot Agent. This showcases the flexibility of the system in integrating various AI services.

```python

```

This example showcases:
1. The use of a Bedrock LLM Agent with Converse API support, allowing for multi-turn conversations.
2. Integration of a Lex Bot Agent for specialized tasks (in this case, travel-related queries).
3. The orchestrator's ability to route requests to the most appropriate agent based on the input.
4. Handling of both streaming and non-streaming responses from different types of agents.


### Working with Anthropic or OpenAI
If you want to use Anthropic or OpenAI for classifier and/or agents, make sure to install the agent-squad with the relevant extra feature.
```bash
pip install "agent-squad[anthropic]"
pip install "agent-squad[openai]"
```

### Full package installation
For a complete installation (including Anthropic and OpenAi):
```bash
pip install agent-squad[all]
```

## Building Locally

This guide explains how to build and install the agent-squad package from source code.

### Prerequisites

- Python 3.11
- pip package manager
- Git (to clone the repository)

### Building the Package

1. Navigate to the Python package directory:
   ```bash
   cd python
   ```

2. Install the build dependencies:
   ```bash
   python -m pip install build
   ```

3. Build the package:
   ```bash
   python -m build
   ```

This process will create distribution files in the `python/dist` directory, including a wheel (`.whl`) file.

### Installation

1. Locate the current version number in `setup.cfg`.

2. Install the built package using pip:
   ```bash
   pip install ./dist/agent_squad-<VERSION>-py3-none-any.whl
   ```
   Replace `<VERSION>` with the version number from `setup.cfg`.

### Example

If the version in `setup.cfg` is `1.2.3`, the installation command would be:
```bash
pip install ./dist/agent_squad-1.2.3-py3-none-any.whl
```

### Troubleshooting

- If you encounter permission errors during installation, you may need to use `sudo` or activate a virtual environment.
- Make sure you're in the correct directory when running the build and install commands.
- Clean the `dist` directory before rebuilding if you encounter issues: `rm -rf python/dist/*`



## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://raw.githubusercontent.com/awslabs/agent-squad/main/CONTRIBUTING.md) for more details.

## 📄 LICENSE

This project is licensed under the Apache 2.0 licence - see the [LICENSE](https://raw.githubusercontent.com/awslabs/agent-squad/main/LICENSE) file for details.

## 📄 Font License
This project uses the JetBrainsMono NF font, licensed under the SIL Open Font License 1.1.
For full license details, see [FONT-LICENSE.md](https://github.com/JetBrains/JetBrainsMono/blob/master/OFL.txt).
