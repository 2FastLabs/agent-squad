<h2 align="center">Agent Squad</h2>
<p align="center">Flexible, lightweight open-source framework for orchestrating multiple AI agents to handle complex conversations.</p>

---
<p align="center">
  <strong>📢 New Name Alert:</strong> Multi-Agent Orchestrator is now <strong>Agent Squad!</strong> 🎉<br>
  Same powerful functionalities, new catchy name. Embrace the squad!
</p>

---

<p align="center">
  <a href="https://github.com/2fastlabs/agent-squad"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repo-green.svg" /></a>
  <a href="https://www.npmjs.com/package/agent-squad"><img alt="npm" src="https://img.shields.io/npm/v/agent-squad.svg?style=flat-square"></a>
  <a href="https://pypi.org/project/agent-squad/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agent-squad.svg?style=flat-square"></a>
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
  <a href="https://2fastlabs.github.io/agent-squad/" style="display: inline-block; background-color: #0066cc; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; font-size: 15px; transition: background-color 0.3s;">
    📚 Explore Full Documentation
  </a>
</p>


## 🔖 Features

- 🧠 **Intelligent intent classification** — Dynamically route queries to the most suitable agent based on context and content.
- 🔤 **Dual language support** — Fully implemented in both **Python** and **TypeScript**.
- 🌊 **Flexible agent responses** — Support for both streaming and non-streaming responses from different agents.
- 📚 **Context management** — Maintain and utilize conversation context across multiple agents for coherent interactions.
- 🔧 **Extensible architecture** — Easily integrate new agents or customize existing ones to fit your specific needs.
- 🌐 **Universal deployment** — Run anywhere - from AWS Lambda to your local environment or any cloud platform.
- 📦 **Pre-built agents and classifiers** — A variety of ready-to-use agents and multiple classifier implementations available.


## What's the Agent Squad ❓

The Agent Squad is a flexible framework for managing multiple AI agents and handling complex conversations. It intelligently routes queries and maintains context across interactions.

The system offers pre-built components for quick deployment, while also allowing easy integration of custom agents and conversation messages storage solutions.

This adaptability makes it suitable for a wide range of applications, from simple chatbots to sophisticated AI systems, accommodating diverse requirements and scaling efficiently.

<hr/>

## 🏗️ High-level architecture flow diagram

<br /><br />

![High-level architecture flow diagram](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/flow.jpg)

<br /><br />

1. The process begins with user input, which is analyzed by a Classifier.
2. The Classifier leverages both Agents' Characteristics and Agents' Conversation history to select the most appropriate agent for the task.
3. Once an agent is selected, it processes the user input.
4. The orchestrator then saves the conversation, updating the Agents' Conversation history, before delivering the response back to the user.


## ![](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/new.png) Introducing SupervisorAgent: Agents Coordination

The Agent Squad now includes a powerful new SupervisorAgent that enables sophisticated team coordination between multiple specialized agents. This new component implements a "agent-as-tools" architecture, allowing a lead agent to coordinate a team of specialized agents in parallel, maintaining context and delivering coherent responses.

![SupervisorAgent flow diagram](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/flow-supervisor.jpg)

Key capabilities:
- 🤝 **Team Coordination** - Coordinate multiple specialized agents working together on complex tasks
- ⚡ **Parallel Processing** - Execute multiple agent queries simultaneously
- 🧠 **Smart Context Management** - Maintain conversation history across all team members
- 🔄 **Dynamic Delegation** - Intelligently distribute subtasks to appropriate team members
- 🤖 **Agent Compatibility** - Works with all agent types (Bedrock, Anthropic, Lex, etc.)

The SupervisorAgent can be used in two powerful ways:
1. **Direct Usage** - Call it directly when you need dedicated team coordination for specific tasks
2. **Classifier Integration** - Add it as an agent within the classifier to build complex hierarchical systems with multiple specialized teams

Here are just a few examples where this agent can be used:
- Customer Support Teams with specialized sub-teams
- AI Movie Production Studios
- Travel Planning Services
- Product Development Teams
- Healthcare Coordination Systems


[Learn more about SupervisorAgent →](https://2fastlabs.github.io/agent-squad/agents/built-in/supervisor-agent)


## 💬 Demo App

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

![](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/demo-app.gif?raw=true)


## 🎯 Examples & Quick Start

Get hands-on experience with the Agent Squad through our diverse set of examples:

- **Demo Applications**:
  - [Streamlit Global Demo](https://github.com/2fastlabs/agent-squad/tree/main/examples/python): A single Streamlit application showcasing multiple demos, including:
    - AI Movie Production Studio
    - AI Travel Planner
  - [Chat Demo App](https://2fastlabs.github.io/agent-squad/cookbook/examples/chat-demo-app/):
    - Explore multiple specialized agents handling various domains like travel, weather, math, and health
  - [E-commerce Support Simulator](https://2fastlabs.github.io/agent-squad/cookbook/examples/ecommerce-support-simulator/): Experience AI-powered customer support with:
    - Automated response generation for common queries
    - Intelligent routing of complex issues to human support
    - Real-time chat and email-style communication
    - Human-in-the-loop interactions for complex cases
- **Sample Projects**: Explore our example implementations in the `examples` folder:
  - [`chat-demo-app`](https://github.com/2fastlabs/agent-squad/tree/main/examples/chat-demo-app): Web-based chat interface with multiple specialized agents
  - [`ecommerce-support-simulator`](https://github.com/2fastlabs/agent-squad/tree/main/examples/ecommerce-support-simulator): AI-powered customer support system
  - [`chat-chainlit-app`](https://github.com/2fastlabs/agent-squad/tree/main/examples/chat-chainlit-app): Chat application built with Chainlit
  - [`fast-api-streaming`](https://github.com/2fastlabs/agent-squad/tree/main/examples/fast-api-streaming): FastAPI implementation with streaming support
  - [`text-2-structured-output`](https://github.com/2fastlabs/agent-squad/tree/main/examples/text-2-structured-output): Natural Language to Structured Data
  - [`bedrock-inline-agents`](https://github.com/2fastlabs/agent-squad/tree/main/examples/bedrock-inline-agents): Bedrock Inline Agents sample
  - [`bedrock-prompt-routing`](https://github.com/2fastlabs/agent-squad/tree/main/examples/bedrock-prompt-routing): Bedrock Prompt Routing sample code


Examples are available in both Python and TypeScript. Check out our [documentation](https://2fastlabs.github.io/agent-squad/) for comprehensive guides on setting up and using the Agent Squad framework!

## 📚 Deep Dives: Stories, Blogs & Podcasts

Discover creative implementations and diverse applications of the Agent Squad:

- **[From 'Bonjour' to 'Boarding Pass': Multilingual AI Chatbot for Flight Reservations](https://community.aws/content/2lCi8jEKydhDm8eE8QFIQ5K23pF/from-bonjour-to-boarding-pass-multilingual-ai-chatbot-for-flight-reservations)**

  This article demonstrates how to build a multilingual chatbot using the Agent Squad framework. The article explains how to use an **Amazon Lex** bot as an agent, along with 2 other new agents to make it work in many languages with just a few lines of code.

- **[Beyond Auto-Replies: Building an AI-Powered E-commerce Support system](https://community.aws/content/2lq6cYYwTYGc7S3Zmz28xZoQNQj/beyond-auto-replies-building-an-ai-powered-e-commerce-support-system)**

  This article demonstrates how to build an AI-driven multi-agent system for automated e-commerce customer email support. It covers the architecture and setup of specialized AI agents using the Agent Squad framework, integrating automated processing with human-in-the-loop oversight. The guide explores email ingestion, intelligent routing, automated response generation, and human verification, providing a comprehensive approach to balancing AI efficiency with human expertise in customer support.

- **[Speak Up, AI: Voicing Your Agents with Amazon Connect, Lex, and Bedrock](https://community.aws/content/2mt7CFG7xg4yw6GRHwH9akhg0oD/speak-up-ai-voicing-your-agents-with-amazon-connect-lex-and-bedrock)**

  This article demonstrates how to build an AI customer call center. It covers the architecture and setup of specialized AI agents using the Agent Squad framework interacting with voice via **Amazon Connect** and **Amazon Lex**.

- **[Unlock Bedrock InvokeInlineAgent API's Hidden Potential](https://community.aws/content/2pTsHrYPqvAbJBl9ht1XxPOSPjR/unlock-bedrock-invokeinlineagent-api-s-hidden-potential-with-agent-squad)**

  Learn how to scale **Amazon Bedrock Agents** beyond knowledge base limitations using the Agent Squad framework and **InvokeInlineAgent API**. This article demonstrates dynamic agent creation and knowledge base selection for enterprise-scale AI applications.

- **[Supercharging Amazon Bedrock Flows](https://community.aws/content/2phMjQ0bqWMg4PBwejBs1uf4YQE/supercharging-amazon-bedrock-flows-with-aws-agent-squad)**

  Learn how to enhance **Amazon Bedrock Flows** with conversation memory and multi-flow orchestration using the Agent Squad framework. This guide shows how to overcome Bedrock Flows' limitations to build more sophisticated AI workflows with persistent memory and intelligent routing between flows.

### 🎙️ Podcast Discussions

- **🇫🇷 Podcast (French)**: L'orchestrateur multi-agents : Un orchestrateur open source pour vos agents IA
  - **Platforms**:
    - [Apple Podcasts](https://podcasts.apple.com/be/podcast/lorchestrateur-multi-agents/id1452118442?i=1000684332612)
    - [Spotify](https://open.spotify.com/episode/4RdMazSRhZUyW2pniG91Vf)


- **🇬🇧 Podcast (English)**: An Orchestrator for Your AI Agents
  - **Platforms**:
    - [Apple Podcasts](https://podcasts.apple.com/us/podcast/an-orchestrator-for-your-ai-agents/id1574162669?i=1000677039579)
    - [Spotify](https://open.spotify.com/episode/2a9DBGZn2lVqVMBLWGipHU)


### TypeScript Version

#### Installation

> 🔄 `multi-agent-orchestrator` becomes `agent-squad`

```bash
npm install agent-squad
```

#### Usage

The following example demonstrates how to use the Agent Squad with two different types of agents: a Bedrock LLM Agent with Converse API support and a Lex Bot Agent. This showcases the flexibility of the system in integrating various AI services.

```typescript
import { AgentSquad, BedrockLLMAgent, LexBotAgent } from "agent-squad";

const orchestrator = new AgentSquad();

// Add a Bedrock LLM Agent with Converse API support
orchestrator.addAgent(
  new BedrockLLMAgent({
      name: "Tech Agent",
      description:
        "Specializes in technology areas including software development, hardware, AI, cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs related to technology products and services.",
      streaming: true
  })
);

// Add a Lex Bot Agent for handling travel-related queries
orchestrator.addAgent(
  new LexBotAgent({
    name: "Travel Agent",
    description: "Helps users book and manage their flight reservations",
    botId: process.env.LEX_BOT_ID,
    botAliasId: process.env.LEX_BOT_ALIAS_ID,
    localeId: "en_US",
  })
);

// Example usage
const response = await orchestrator.routeRequest(
  "I want to book a flight",
  'user123',
  'session456'
);

// Handle the response (streaming or non-streaming)
if (response.streaming == true) {
    console.log("\n** RESPONSE STREAMING ** \n");
    // Send metadata immediately
    console.log(`> Agent ID: ${response.metadata.agentId}`);
    console.log(`> Agent Name: ${response.metadata.agentName}`);
    console.log(`> User Input: ${response.metadata.userInput}`);
    console.log(`> User ID: ${response.metadata.userId}`);
    console.log(`> Session ID: ${response.metadata.sessionId}`);
    console.log(
      `> Additional Parameters:`,
      response.metadata.additionalParams
    );
    console.log(`\n> Response: `);

    // Stream the content
    for await (const chunk of response.output) {
      if (typeof chunk === "string") {
        process.stdout.write(chunk);
      } else {
        console.error("Received unexpected chunk type:", typeof chunk);
      }
    }

} else {
    // Handle non-streaming response (AgentProcessingResult)
    console.log("\n** RESPONSE ** \n");
    console.log(`> Agent ID: ${response.metadata.agentId}`);
    console.log(`> Agent Name: ${response.metadata.agentName}`);
    console.log(`> User Input: ${response.metadata.userInput}`);
    console.log(`> User ID: ${response.metadata.userId}`);
    console.log(`> Session ID: ${response.metadata.sessionId}`);
    console.log(
      `> Additional Parameters:`,
      response.metadata.additionalParams
    );
    console.log(`\n> Response: ${response.output}`);
}
```

### Python Version

> 🔄 `multi-agent-orchestrator` becomes `agent-squad`

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
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, AgentStreamResponse

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

These examples showcase:
1. The use of a Bedrock LLM Agent with Converse API support, allowing for multi-turn conversations.
2. Integration of a Lex Bot Agent for specialized tasks (in this case, travel-related queries).
3. The orchestrator's ability to route requests to the most appropriate agent based on the input.
4. Handling of both streaming and non-streaming responses from different types of agents.


### Modular Installation Options

The Agent Squad is designed with a modular architecture, allowing you to install only the components you need while ensuring you always get the core functionality.

#### Installation Options

**1. AWS Integration**:

  ```bash
   pip install "agent-squad[aws]"
  ```
Includes core orchestration functionality with comprehensive AWS service integrations (`BedrockLLMAgent`, `AmazonBedrockAgent`, `LambdaAgent`, etc.)

**2. Anthropic Integration**:

  ```bash
pip install "agent-squad[anthropic]"
  ```

**3. OpenAI Integration**:

  ```bash
pip install "agent-squad[openai]"
  ```

Adds OpenAI's GPT models for agents and classification, along with core packages.

**4. Full Installation**:

  ```bash
pip install "agent-squad[all]"
  ```

Includes all optional dependencies for maximum flexibility.


### 🙌 **We Want to Hear From You!**

Have something to share, discuss, or brainstorm? We’d love to connect with you and hear about your journey with the **Agent Squad framework**. Here’s how you can get involved:

- **🙌 Show & Tell**: Got a success story, cool project, or creative implementation? Share it with us in the [**Show and Tell**](https://github.com/2fastlabs/agent-squad/discussions/categories/show-and-tell) section. Your work might inspire the entire community! 🎉

- **💬 General Discussion**: Have questions, feedback, or suggestions? Join the conversation in our [**General Discussions**](https://github.com/2fastlabs/agent-squad/discussions/categories/general) section. It’s the perfect place to connect with other users and contributors.

- **💡 Ideas**: Thinking of a new feature or improvement? Share your thoughts in the [**Ideas**](https://github.com/2fastlabs/agent-squad/discussions/categories/ideas) section. We’re always open to exploring innovative ways to make the orchestrator even better!

Let’s collaborate, learn from each other, and build something incredible together! 🚀

## 📝 Pull Request Guidelines

### Issue-First Policy

This repository follows an **Issue-First** policy:

- **Every pull request must be linked to an existing issue**
- If there isn't an issue for the changes you want to make, please create one first
- Use the issue to discuss proposed changes before investing time in implementation

### How to Link Pull Requests to Issues

When creating a pull request, you must link it to an issue using one of these methods:

1. Include a reference in the PR description using keywords:
   - `Fixes #123`
   - `Resolves #123`
   - `Closes #123`

2. Manually link the PR to an issue through GitHub's UI:
   - On the right sidebar of your PR, click "Development" and then "Link an issue"

### Automated Enforcement

We use GitHub Actions to automatically verify that each PR is linked to an issue. PRs without linked issues will not pass required checks and cannot be merged.

This policy helps us:
- Maintain clear documentation of changes and their purposes
- Ensure community discussion before implementation
- Keep a structured development process
- Make project history more traceable and understandable

## 🤝 Contributing

⚠️ Note: Our project has been renamed from **Multi-Agent Orchestrator** to **Agent Squad**. Please use the new name in your contributions and discussions.

⚠️ We value your contributions! Before submitting changes, please start a discussion by opening an issue to share your proposal.

Once your proposal is approved, here are the next steps:

1. 📚 Review our [Contributing Guide](CONTRIBUTING.md)
2. 💡 Create a [GitHub Issue](https://github.com/2fastlabs/agent-squad/issues)
3. 🔨 Submit a pull request


✅ Follow existing project structure and include documentation for new features.


🌟 **Stay Updated**: Star the repository to be notified about new features, improvements, and exciting developments in the Agent Squad framework!

# Authors

- [Corneliu Croitoru](https://www.linkedin.com/in/corneliucroitoru/)
- [Anthony Bernabeu](https://www.linkedin.com/in/anthonybernabeu/)

# 👥 Contributors

Big shout out to our awesome contributors! Thank you for making this project better! 🌟 ⭐ 🚀

[![contributors](https://contrib.rocks/image?repo=2fastlabs/agent-squad&max=2000)](https://github.com/2fastlabs/agent-squad/graphs/contributors)


Please see our [contributing guide](./CONTRIBUTING.md) for guidelines on how to propose bugfixes and improvements.


## 📄 LICENSE

This project is licensed under the Apache 2.0 licence - see the [LICENSE](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/LICENSE) file for details.

## 📄 Font License
This project uses the JetBrainsMono NF font, licensed under the SIL Open Font License 1.1.
For full license details, see [FONT-LICENSE.md](https://github.com/JetBrains/JetBrainsMono/blob/master/OFL.txt).
