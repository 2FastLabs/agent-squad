---
name: agent-squad-python
description: >-
  Use when building or modifying a Python app that uses the agent-squad Python package — async
  multi-agent orchestration for Python 3.11+: orchestrator, agents (BedrockLLMAgent,
  AnthropicAgent, OpenAIAgent, SupervisorAgent, GroundedAgent, ChainAgent, and more), classifier
  routing (Bedrock, Anthropic, OpenAI), storage (in-memory, DynamoDB, SQL/Turso), retrievers
  (Amazon KB, Dakera), tools (AgentTools, MCPToolProvider), and custom implementations.
---

# agent-squad Python — assistant guide

Async-first, dependency-optional multi-agent orchestration framework (Python 3.11+). This is a
guide and a map — **not an API reference**. Read exact signatures from the source
(`python/src/agent_squad/`) and the docs site (`docs/src/content/docs/`); this file tells you
*what to use, when, and what to watch out for*.

## When to use what

- **One assistant** — a single `Agent` subclass; no orchestrator needed, call `process_request`
  directly.
- **Several specialists** — multiple agents + an `AgentSquad` orchestrator; the classifier routes
  each turn to the right agent automatically.
- **Answers must not drift from data** (prices, balances, live stock) — `GroundedAgent`: a gatherer
  LLM calls tools and sees raw results but never speaks to the user; an isolated presenter LLM
  writes the reply grounded only on curated facts.
- **Fixed pipeline** — `ChainAgent`: routes the output of one agent as the input to the next,
  sequentially.
- **Team coordination** — `SupervisorAgent`: a lead `BedrockLLMAgent` or `AnthropicAgent` delegates
  to a team of sub-agents via an internal tool loop, maintaining shared context. Can itself be
  registered in an `AgentSquad`.
- **External tool servers** — `MCPToolProvider` (requires `agent-squad[mcp]`) connects any number
  of MCP servers (stdio or SSE) and makes their tools available to any agent.

## How to install

All third-party integrations are optional extras — never forced on users who don't need them.

```bash
pip install agent-squad                  # core only (no LLM runtime)
pip install "agent-squad[aws]"           # + boto3 — BedrockLLMAgent, BedrockClassifier, DynamoDbChatStorage, etc.
pip install "agent-squad[anthropic]"     # + anthropic SDK — AnthropicAgent, AnthropicClassifier
pip install "agent-squad[openai]"        # + openai SDK — OpenAIAgent, OpenAIClassifier
pip install "agent-squad[sql]"           # + libsql-client — SqlChatStorage (Turso/libSQL)
pip install "agent-squad[strands-agents]"# + strands-agents — StrandsAgent
pip install "agent-squad[dakera]"        # + dakera — DakeraRetriever
pip install "agent-squad[mcp]"           # + mcp — MCPToolProvider
pip install "agent-squad[all]"           # everything except strands-agents
```

## How a turn works

`AgentSquad.route_request` is the one entry point worth memorising. It is a coroutine — you must
`await` it.

```python
import asyncio
from agent_squad.orchestrator import AgentSquad
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions

orchestrator = AgentSquad(
    classifier=BedrockClassifier(BedrockClassifierOptions())
)
orchestrator.add_agent(BedrockLLMAgent(BedrockLLMAgentOptions(
    name="General Assistant",
    description="Handles general knowledge questions",
)))

async def main():
    response = await orchestrator.route_request(
        user_input="What is the capital of France?",
        user_id="user-123",
        session_id="session-abc",
    )

    if response.streaming:
        # response.output is an async generator of AgentStreamResponse
        async for chunk in response.output:
            if chunk.text:
                print(chunk.text, end="", flush=True)
            if chunk.final_message:
                pass  # full ConversationMessage — already persisted
    else:
        # response.output is a ConversationMessage
        print(response.output.content[0]["text"])

asyncio.run(main())
```

`AgentResponse` has three fields: `metadata` (`AgentProcessingResult`), `output`, and `streaming`
(bool). Always branch on `response.streaming` — the type of `output` differs.

To stream back from `route_request`, pass `stream_response=True`:

```python
response = await orchestrator.route_request(
    user_input="...",
    user_id="u1",
    session_id="s1",
    stream_response=True,
)
```

If no agent is selected and no default agent is configured, `route_request` returns an
`AgentResponse` with the `NO_SELECTED_AGENT_MESSAGE` text rather than raising.

## The pieces

### AgentSquad (orchestrator)

`from agent_squad.orchestrator import AgentSquad`

The top-level object. Holds an agent registry, a classifier, and a `ChatStorage`.

```python
from agent_squad.types import AgentSquadConfig

orchestrator = AgentSquad(
    options=AgentSquadConfig(
        LOG_CLASSIFIER_OUTPUT=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=20,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    ),
    storage=my_storage,        # default: InMemoryChatStorage
    classifier=my_classifier,  # default: BedrockClassifier (if boto3 installed)
    default_agent=fallback,    # used when classifier returns no match
)
orchestrator.add_agent(agent)
```

`AgentSquadConfig` fields: `LOG_AGENT_CHAT`, `LOG_CLASSIFIER_CHAT`, `LOG_CLASSIFIER_RAW_OUTPUT`,
`LOG_CLASSIFIER_OUTPUT`, `LOG_EXECUTION_TIMES`, `MAX_RETRIES`, `USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED`,
`NO_SELECTED_AGENT_MESSAGE`, `GENERAL_ROUTING_ERROR_MSG_MESSAGE`, `MAX_MESSAGE_PAIRS_PER_AGENT`.

You can also call `classify_request` and `agent_process_request` separately if you need to inspect
the routing decision before dispatching.

### Agents

All agents require `agent-squad[aws]`, `[anthropic]`, or `[openai]` depending on the underlying
SDK. The base `Agent` and `AgentOptions` plus `SupervisorAgent` and `GroundedAgent` are always
available with the core install.

| Agent | Extra needed | Notes |
|---|---|---|
| `BedrockLLMAgent` | `aws` | Bedrock Converse API; supports streaming, tools, retriever |
| `AmazonBedrockAgent` | `aws` | Bedrock Agents runtime (managed agents with KB/action groups) |
| `BedrockInlineAgent` | `aws` | Bedrock inline agents — code interpretation, KB, and tools inline |
| `BedrockFlowsAgent` | `aws` | Bedrock Flows — runs a preconfigured flow |
| `BedrockTranslatorAgent` | `aws` | Bedrock translation agent |
| `LambdaAgent` | `aws` | Invokes an AWS Lambda function |
| `LexBotAgent` | `aws` | Routes to an Amazon Lex bot |
| `ComprehendFilterAgent` | `aws` | Comprehend PII/toxicity filter before passing to another agent |
| `ChainAgent` | `aws` | Sequential pipeline — each agent's output feeds the next |
| `AnthropicAgent` | `anthropic` | Anthropic Messages API; supports streaming and tools |
| `OpenAIAgent` | `openai` | OpenAI Chat Completions API; supports streaming and tools |
| `StrandsAgent` | `strands-agents` | Strands Agents integration |
| `SupervisorAgent` | `aws` or `anthropic` | Lead agent coordinates a team via tools; always available in `__init__.py` but requires a compatible `lead_agent` |
| `GroundedAgent` | same as gatherer/presenter | Two-LLM anti-hallucination pattern; always importable |

All agents extend `Agent` (`agent_squad.agents.agent`). The `AgentOptions` dataclass is the
construction pattern — every concrete agent has a matching `*Options` dataclass that extends it:

```python
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from agent_squad.utils import AgentTools, AgentTool

agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="My Agent",
    description="Handles X",              # shown to the classifier
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    streaming=True,
    save_chat=True,                        # default True — persists history
    tool_config={"tool": my_tools},        # AgentTools instance
    retriever=my_retriever,
    LOG_AGENT_DEBUG_TRACE=False,
))
```

**`GroundedAgent`** takes a `gatherer`, `presenter`, `tools`, optional `curator`
(`ToolOutputCurator`), and optional `presenter_prompt` (`PresenterPrompt`). The gatherer runs the
tool loop; the presenter receives only the curated data and never sees the chat history or the
gatherer's transcript.

**`SupervisorAgent`** takes a `lead_agent` (must be `BedrockLLMAgent` or `AnthropicAgent`), a
`team` list, optional `storage`, and optional `extra_tools`. The lead agent must not have its own
`tool_config` — tools are managed internally. Use `extra_tools` for additional tools beyond the
team-dispatch tools.

**`ChainAgent`** (requires `aws`) takes an `agents` list. Each agent's text output becomes the next
agent's input.

### Classifiers

`from agent_squad.classifiers import BedrockClassifier, AnthropicClassifier, OpenAIClassifier`

| Classifier | Extra needed |
|---|---|
| `BedrockClassifier` | `aws` |
| `AnthropicClassifier` | `anthropic` |
| `OpenAIClassifier` | `openai` |

The orchestrator defaults to `BedrockClassifier` if `boto3` is installed and no classifier is
provided. If boto3 is not installed and no classifier is passed, the orchestrator raises
`ValueError` at construction time.

All classifiers extend `Classifier` (`agent_squad.classifiers.classifier`). You can override the
routing prompt via `set_system_prompt(template, variables)` where `{{AGENT_DESCRIPTIONS}}` and
`{{HISTORY}}` are the built-in template placeholders.

### Storage

`from agent_squad.storage import InMemoryChatStorage, DynamoDbChatStorage, SqlChatStorage`

| Storage | Extra needed | Notes |
|---|---|---|
| `InMemoryChatStorage` | none | Default; not persistent |
| `DynamoDbChatStorage` | `aws` | DynamoDB-backed; production default for AWS deployments |
| `SqlChatStorage` | `sql` | libSQL/Turso-backed |
| `SummarizingChatStorage` | none | Wraps any storage; compresses history via a user-supplied async callable when history exceeds `trigger_at` pairs |

All storage classes extend `ChatStorage` (`agent_squad.storage.chat_storage`). Storage is keyed by
`(user_id, session_id, agent_id)`. `MAX_MESSAGE_PAIRS_PER_AGENT` (default 100) trims history at
write time. `save_chat=False` on an agent disables history for that agent only.

### Retrievers

`from agent_squad.retrievers import AmazonKnowledgeBasesRetriever, DakeraRetriever`

| Retriever | Extra needed | Notes |
|---|---|---|
| `AmazonKnowledgeBasesRetriever` | `aws` | Amazon Bedrock Knowledge Bases |
| `DakeraRetriever` | `dakera` | Self-hosted Dakera memory server |

All retrievers extend `Retriever` (`agent_squad.retrievers.retriever`). A retriever attached to an
agent augments its prompt with retrieved context before the LLM call.

`DakeraRetriever` reads `DAKERA_API_KEY` and `DAKERA_URL` from environment variables if not
provided in `DakeraRetrieverOptions`.

### Tools

**`AgentTools` / `AgentTool`** — the native tool system, always available:

```python
from agent_squad.utils import AgentTools, AgentTool

def get_weather(city: str) -> str:
    """Get the weather for a city.
    :param city: The city name.
    """
    return f"Sunny in {city}"

tools = AgentTools(tools=[
    AgentTool(name="get_weather", func=get_weather)
])
# AgentTool auto-extracts properties from type hints and :param docstrings.
```

`AgentTool` wraps both sync and async functions transparently. Properties, descriptions, and
required fields can be overridden explicitly. `AgentTools.tool_handler` processes tool call
responses for Bedrock and Anthropic wire formats.

**`MCPToolProvider`** — drop-in `AgentTools` subclass for MCP servers (requires `agent-squad[mcp]`):

```python
from agent_squad.tools import MCPToolProvider, MCPServerConfig

provider = await MCPToolProvider.create([
    MCPServerConfig(type="stdio", command="uvx", args=["my-mcp-server"]),
    MCPServerConfig(type="sse", url="http://localhost:3000/sse"),
])

agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="MCP Agent",
    description="...",
    tool_config={"tool": provider},
))

# Clean up when done:
await provider.disconnect()
```

`MCPToolProvider.create` is an async factory — it connects to all servers and fetches tool
definitions upfront so they are available synchronously when the agent builds its API request.

### Callbacks

`AgentCallbacks` (on `AgentOptions`) hooks into the agent lifecycle:
- `on_agent_start` — returns a dict (tracking info) available to other callbacks via `kwargs`
- `on_agent_end`
- `on_llm_start`
- `on_llm_new_token`
- `on_llm_end`

`AgentToolCallbacks` (on `AgentTools`) hooks into tool execution:
- `on_tool_start`
- `on_tool_end`
- `on_tool_error`

`ClassifierCallbacks` hooks into the classifier:
- `on_classifier_start`
- `on_classifier_stop`

## Custom implementations

Subclass the abstract base and pass your instance where the built-in goes. Source paths are
under `python/src/agent_squad/`.

| Seam | Base class | Source file |
|---|---|---|
| Agent | `Agent` | `agents/agent.py` |
| Classifier | `Classifier` | `classifiers/classifier.py` |
| Storage | `ChatStorage` | `storage/chat_storage.py` |
| Retriever | `Retriever` | `retrievers/retriever.py` |
| Tool curator | `ToolOutputCurator` | `agents/grounded_agent.py` |
| Presenter prompt | `PresenterPrompt` | `agents/grounded_agent.py` |

Minimal custom agent:

```python
from typing import Optional, AsyncIterable, Union
from agent_squad.agents import Agent, AgentOptions
from agent_squad.types import ConversationMessage, ParticipantRole

class MyAgent(Agent):
    def __init__(self, options: AgentOptions):
        super().__init__(options)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict] = None,
    ) -> Union[ConversationMessage, AsyncIterable]:
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Echo: {input_text}"}],
        )
```

For a streaming custom agent, also override `is_streaming_enabled` to return `True` and yield
`AgentStreamResponse` objects (set `final_message` on the last one).

## Gotchas

- **All public methods are async.** `route_request`, `process_request`, `classify`, storage methods,
  and retriever methods are all coroutines. You must `await` them inside an async context. Use
  `asyncio.run(main())` at the top level.
- **Optional imports at module level, not inside methods.** The framework guards all optional
  integrations with `try/except ImportError` at the top of each module. If you copy this pattern
  for your own extensions, keep the guard at module level — never inside `__init__` or a method.
- **Classifier is required.** Unlike the Swift version, the Python `AgentSquad` raises at
  construction time if no classifier can be resolved (no `boto3` installed and no classifier
  passed). Always pass a `classifier` explicitly when `boto3` is not available.
- **`response.streaming` and `response.output` type are coupled.** When `streaming=True`, `output`
  is an async generator; when `False`, it is a `ConversationMessage`. Always branch on
  `response.streaming` before consuming `output`.
- **`stream_response=False` by default.** Even if the agent itself streams internally, the
  orchestrator will drain the stream and return a single `ConversationMessage` unless you pass
  `stream_response=True` to `route_request`.
- **Agent `id` is derived from `name`** via `generate_key_from_name`: lowercased, spaces replaced
  with hyphens, special characters stripped. `agent.id` is the storage key — two agents with names
  that normalise to the same string will collide. Pick distinct names.
- **`AgentOptions` is a dataclass; new fields must have defaults.** When subclassing (e.g.
  `BedrockLLMAgentOptions`), add new fields with defaults so existing construction call-sites
  keep working.
- **`SupervisorAgent` name and description come from the lead agent.** Whatever you set on
  `SupervisorAgentOptions.name` / `.description` is overwritten by `lead_agent.name` /
  `lead_agent.description` at construction time.
- **`SupervisorAgent` forbids `tool_config` on the lead agent.** The supervisor manages tools
  internally. Use `extra_tools` for any additional tools beyond team dispatch.
- **`GroundedAgent` presenter isolation is strict.** The presenter never sees chat history or the
  gatherer's transcript — only the curated data produced by the `ToolOutputCurator`. A chit-chat
  turn that calls no tools is answered by the gatherer alone (presenter is skipped).
- **`MCPToolProvider.create` is async.** It must be `await`ed before building the agent. Call
  `await provider.disconnect()` when done to close server connections.
- **`save_chat=True` by default.** Every agent persists both sides of each exchange unless
  explicitly set to `False`. Storage is scoped per `(user_id, session_id, agent_id)`. The
  `MAX_MESSAGE_PAIRS_PER_AGENT` config trims at save time, not at fetch time.
- **`ConversationMessage.content` is a list of dicts**, not a plain string. Text is at
  `content[0]["text"]` for most agents. Some agents may produce multi-block content (tool use,
  images). Do not assume `content` has a single element.
- **`DakeraRetriever` and `AmazonKnowledgeBasesRetriever` are not import-guarded** — they are
  always exported from `agent_squad.retrievers`. If the underlying SDK (`dakera`, `boto3`) is not
  installed, the import will fail at runtime when you try to construct them.

## Go deeper

- **Prose and recipes** — the Starlight docs under `docs/src/content/docs/` (run with
  `npm run dev` from `docs/`): `get-started/`, `agents/`, `classifiers/`, `storage/`, `retrievers/`,
  `tools/`.
- **Exact signatures** — `python/src/agent_squad/`:
  - `orchestrator.py` — `AgentSquad`, `route_request`, `classify_request`, `agent_process_request`
  - `agents/agent.py` — `Agent`, `AgentOptions`, `AgentCallbacks`, `AgentResponse`, `AgentStreamResponse`
  - `agents/grounded_agent.py` — `GroundedAgent`, `GroundedAgentOptions`, `ToolOutputCurator`, `DataBlockCurator`, `PerToolCurator`, `PresenterPrompt`, `CapturedToolResult`
  - `agents/supervisor_agent.py` — `SupervisorAgent`, `SupervisorAgentOptions`
  - `agents/chain_agent.py` — `ChainAgent`, `ChainAgentOptions`
  - `classifiers/classifier.py` — `Classifier`, `ClassifierResult`, `ClassifierCallbacks`
  - `storage/chat_storage.py` — `ChatStorage`
  - `retrievers/retriever.py` — `Retriever`
  - `utils/tool.py` — `AgentTools`, `AgentTool`, `AgentToolCallbacks`, `AgentToolResult`
  - `tools/mcp_tool_provider.py` — `MCPToolProvider`, `MCPServerConfig`
  - `types/types.py` — `ConversationMessage`, `ParticipantRole`, `AgentSquadConfig`, `TimestampedMessage`
- **Tests** — `python/src/tests/` — pytest; run from `python/` with `make test`.
- **Optional extras** — `python/setup.cfg` — `[options.extras_require]`.
