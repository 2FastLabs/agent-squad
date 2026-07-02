<h2 align="center">Agent Squad</h2>
<p align="center">Flexible, lightweight open-source framework for orchestrating multiple AI agents — in the cloud with Python and TypeScript, and now <strong>on device</strong> with Swift.</p>

<p align="center">
  <a href="https://www.npmjs.com/package/agent-squad"><img alt="npm" src="https://img.shields.io/npm/v/agent-squad.svg?style=flat-square"></a>
  <a href="https://pypi.org/project/agent-squad/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agent-squad.svg?style=flat-square"></a>
  <a href="swift/README.md"><img alt="Swift" src="https://img.shields.io/badge/Swift-iOS%2016%20%C2%B7%20macOS%2014-orange.svg?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square"></a>
</p>

<h3 align="center">
  <img src="https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/new.png" alt="New" height="20"> Now available in Swift
</h3>
<p align="center">
  On-device agent orchestration for iPhone, iPad, and Mac — agents, MCP tools, realtime voice, and tracing, running entirely on device.<br>
  <a href="#new-the-swift-runtime--on-device-orchestration"><strong>See what's included ↓</strong></a> · <a href="swift/README.md"><strong>Swift README →</strong></a>
</p>

<p align="center">
  <a href="https://2fastlabs.github.io/agent-squad/"><strong>Explore the full documentation</strong></a>
</p>

> **New home:** previously hosted at `awslabs/agent-squad`, the project is now maintained at `2fastlabs/agent-squad` (and was formerly named `multi-agent-orchestrator`). Please update bookmarks, clone URLs, and dependencies.

## What is Agent Squad?

Agent Squad routes each user query to the most suitable of your specialized agents and maintains conversation context across them. You get pre-built agents, classifiers, and storage for quick deployment, plus small, well-defined seams to plug in your own.

- 🧠 **Intelligent intent classification** — route queries by context and content.
- 🌊 **Streaming and non-streaming** agent responses.
- 📚 **Context management** — coherent conversations across agents and sessions.
- 🔧 **Extensible by design** — custom agents, classifiers, storage, and retrievers.
- 📦 **Pre-built agents** — Bedrock, Anthropic, OpenAI, Lex, Lambda, and more.

## Three runtimes, one framework

| Runtime | Requirements |
|---|---|
| [**Python**](#python) | Python 3.11+ |
| [**TypeScript**](#typescript) | Node.js |
| [**Swift**](#swift) — **new** | iOS 16+ / macOS 14+ |

Python and TypeScript maintain feature parity and run anywhere — AWS Lambda, containers, your laptop. The new **Swift runtime** brings the same orchestration model to Apple platforms and runs **entirely on device**: classifier routing, tools (native and [MCP](https://modelcontextprotocol.io)), realtime voice, tracing, and local-first chat storage.

## How it works

![High-level architecture flow diagram](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/flow.jpg)

1. User input is analyzed by a **Classifier**.
2. The Classifier uses the agents' descriptions and the conversation history to select the best agent for the turn.
3. The selected **Agent** processes the input (calling tools as needed).
4. The **Orchestrator** saves the exchange and returns the response.

## Quick start

### TypeScript

```bash
npm install agent-squad
```

```typescript
import { AgentSquad, BedrockLLMAgent } from "agent-squad";

const orchestrator = new AgentSquad();

orchestrator.addAgent(
  new BedrockLLMAgent({
    name: "Tech Agent",
    description: "Specializes in technology: software, hardware, AI, cybersecurity, cloud.",
    streaming: true
  })
);

const response = await orchestrator.routeRequest("What is AWS Lambda?", "user123", "session456");

console.log(`> Agent: ${response.metadata.agentName}\n`);
if (response.streaming) {
  for await (const chunk of response.output) {
    if (typeof chunk === "string") process.stdout.write(chunk);
  }
} else {
  console.log(response.output);
}
```

### Python

```bash
pip install "agent-squad[aws]"   # or [anthropic], [openai], [all] — see the docs
```

```python
import asyncio
from agent_squad.orchestrator import AgentSquad
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, AgentStreamResponse

orchestrator = AgentSquad()
orchestrator.add_agent(BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Tech Agent",
    description="Specializes in technology: software, hardware, AI, cybersecurity, cloud.",
    streaming=True,
)))

async def main():
    response = await orchestrator.route_request("What is AWS Lambda?", "user123", "session456", {}, True)

    print(f"> Agent: {response.metadata.agent_name}\n")
    if response.streaming:
        async for chunk in response.output:
            if isinstance(chunk, AgentStreamResponse):
                print(chunk.text, end="", flush=True)
    else:
        print(response.output.content)

asyncio.run(main())
```

### Swift

Add the package to your `Package.swift` (or via Xcode → Add Package Dependencies):

```swift
dependencies: [
    .package(url: "https://github.com/2FastLabs/agent-squad", branch: "main")
]
```

```swift
import AgentSquad

let agent = Agent(name: "Shop", description: "Shopping assistant",
                  model: ChatCompletionsClient(model: "gpt-4o-mini", apiKey: apiKey))
let orchestrator = Orchestrator(agents: [agent], store: try DeviceChatStorage(userId: "u1"))

for try await event in orchestrator.route(.text("wireless headphones under €100?"),
                                          userId: "u1", sessionId: "s1") {
    if case .textDelta(let token) = event { print(token, terminator: "") }
}
```

Full walkthrough in the [Swift README](swift/README.md#quick-start).

## SupervisorAgent — team coordination

A lead agent coordinates a team of specialized agents in parallel using an *agent-as-tools* architecture, maintaining shared context and delivering one coherent response.

![SupervisorAgent flow diagram](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/flow-supervisor.jpg)

- **Team coordination** with **parallel** sub-agent queries and smart shared context.
- **Dynamic delegation** of subtasks to the right team member.
- Works with **all agent types** — and can itself be registered in the classifier to build hierarchical teams of teams.

[Learn more about SupervisorAgent →](https://2fastlabs.github.io/agent-squad/agents/built-in/supervisor-agent)

## GroundedAgent — answers that can't drift from your data

Available in **all three runtimes**, `GroundedAgent` is the framework's anti-hallucination pattern: two LLMs instead of one.

![GroundedAgent flow diagram](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/docs/public/grounded-agent.png)

- A **gatherer** calls your tools and sees the raw results — but never speaks to the user.
- An isolated **presenter** writes the reply from the curated tool output alone: no tools, no tool transcript, no chat history. It cannot invent a price, a rating, or a stock status that wasn't actually fetched.
- A no-tool turn skips the presenter and answers in one pass.

Use it wherever answers must match the data exactly — prices, odds, balances, availability.

[Learn more about GroundedAgent →](https://2fastlabs.github.io/agent-squad/agents/built-in/grounded-agent)

## New: the Swift runtime — on-device orchestration

The orchestration model above, rebuilt for Apple platforms as a protocol-driven Swift 6 package — designed to run the whole loop **on device**:

- 🧠 **Swappable agents** — `Agent`, `GroundedAgent`, or your own `AgentProtocol` conformance, routed by an optional `LLMClassifier`.
- 🧰 **Tools from any source** — native Swift functions, declarative HTTP tools, or any **MCP server**, composed behind one seam.
- 🎙️ **Realtime voice** — natural spoken conversations with interrupt-to-speak, sharing the same grounded gatherer → presenter core.
- 📈 **First-class tracing** — OSLog during development, OTLP export (Langfuse, LangSmith, Datadog, …) in production.
- 💾 **Local-first chat history** — JSON-file or SwiftData persistence on device, swappable like everything else.

Start with the [Swift README](swift/README.md) and the [Swift docs](https://2fastlabs.github.io/agent-squad/swift/quick-start/).

## Examples & demos

Watch the demo app route a conversation across six specialized agents (travel, weather, restaurants, math, tech, health) while preserving context through brief follow-ups:

![Demo app](https://raw.githubusercontent.com/2fastlabs/agent-squad/main/img/demo-app.gif?raw=true)

- [Streamlit Global Demo](https://github.com/2fastlabs/agent-squad/tree/main/examples/python) — AI Movie Production Studio, AI Travel Planner, and more in one app.
- [`chat-demo-app`](https://github.com/2fastlabs/agent-squad/tree/main/examples/chat-demo-app) — web chat interface with multiple specialized agents ([guide](https://2fastlabs.github.io/agent-squad/cookbook/examples/chat-demo-app/)).
- [`ecommerce-support-simulator`](https://github.com/2fastlabs/agent-squad/tree/main/examples/ecommerce-support-simulator) — AI-powered customer support with human-in-the-loop ([guide](https://2fastlabs.github.io/agent-squad/cookbook/examples/ecommerce-support-simulator/)).
- [`chat-chainlit-app`](https://github.com/2fastlabs/agent-squad/tree/main/examples/chat-chainlit-app) — chat application built with Chainlit.
- [`fast-api-streaming`](https://github.com/2fastlabs/agent-squad/tree/main/examples/fast-api-streaming) — FastAPI with streaming.
- [`text-2-structured-output`](https://github.com/2fastlabs/agent-squad/tree/main/examples/text-2-structured-output) — natural language to structured data.
- [`bedrock-inline-agents`](https://github.com/2fastlabs/agent-squad/tree/main/examples/bedrock-inline-agents) · [`bedrock-prompt-routing`](https://github.com/2fastlabs/agent-squad/tree/main/examples/bedrock-prompt-routing) — Bedrock samples.

## Articles & podcasts

- [Multilingual AI chatbot for flight reservations](https://community.aws/content/2lCi8jEKydhDm8eE8QFIQ5K23pF/from-bonjour-to-boarding-pass-multilingual-ai-chatbot-for-flight-reservations) — Amazon Lex as an agent, many languages in a few lines.
- [Building an AI-powered e-commerce support system](https://community.aws/content/2lq6cYYwTYGc7S3Zmz28xZoQNQj/beyond-auto-replies-building-an-ai-powered-e-commerce-support-system) — email ingestion, routing, and human verification.
- [Voicing your agents with Amazon Connect, Lex, and Bedrock](https://community.aws/content/2mt7CFG7xg4yw6GRHwH9akhg0oD/speak-up-ai-voicing-your-agents-with-amazon-connect-lex-and-bedrock) — an AI customer call center.
- [Unlock Bedrock InvokeInlineAgent API's hidden potential](https://community.aws/content/2pTsHrYPqvAbJBl9ht1XxPOSPjR/unlock-bedrock-invokeinlineagent-api-s-hidden-potential-with-agent-squad) — dynamic agent creation at enterprise scale.
- [Supercharging Amazon Bedrock Flows](https://community.aws/content/2phMjQ0bqWMg4PBwejBs1uf4YQE/supercharging-amazon-bedrock-flows-with-aws-agent-squad) — conversation memory and multi-flow orchestration.
- **Podcasts**: [An Orchestrator for Your AI Agents (EN)](https://podcasts.apple.com/us/podcast/an-orchestrator-for-your-ai-agents/id1574162669?i=1000677039579) ([Spotify](https://open.spotify.com/episode/2a9DBGZn2lVqVMBLWGipHU)) · [L'orchestrateur multi-agents (FR)](https://podcasts.apple.com/be/podcast/lorchestrateur-multi-agents/id1452118442?i=1000684332612) ([Spotify](https://open.spotify.com/episode/4RdMazSRhZUyW2pniG91Vf))

## Community & contributing

Questions, ideas, or something to show off? Join the [discussions](https://github.com/2fastlabs/agent-squad/discussions): [Show & Tell](https://github.com/2fastlabs/agent-squad/discussions/categories/show-and-tell) · [General](https://github.com/2fastlabs/agent-squad/discussions/categories/general) · [Ideas](https://github.com/2fastlabs/agent-squad/discussions/categories/ideas).

Contributions are welcome. This repository follows an **issue-first policy**: every pull request must be linked to an issue (`Fixes #123` in the PR body, or GitHub's "Link an issue"); a required CI check enforces it. Open an issue to discuss your proposal, then see the [Contributing Guide](CONTRIBUTING.md) for build and test instructions per runtime.

Star the repository to be notified about new features and releases.

## Authors

- [Corneliu Croitoru](https://www.linkedin.com/in/corneliucroitoru/)
- [Anthony Bernabeu](https://www.linkedin.com/in/anthonybernabeu/)

## Contributors

Big shout out to our awesome contributors! Thank you for making this project better!

[![contributors](https://contrib.rocks/image?repo=2fastlabs/agent-squad&max=2000)](https://github.com/2fastlabs/agent-squad/graphs/contributors)

## Support Agent Squad

If Agent Squad has helped you or your organization build AI applications faster, consider [sponsoring its development](https://github.com/sponsors/2FastLabs). Your sponsorship funds maintenance, documentation, and new features — keeping the project healthy for the entire community.

## License

This project is licensed under the Apache 2.0 license — see the [LICENSE](LICENSE) file for details.

This project uses the JetBrainsMono NF font, licensed under the [SIL Open Font License 1.1](https://github.com/JetBrains/JetBrainsMono/blob/master/OFL.txt).
