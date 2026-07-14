---
name: agent-squad-typescript
description: >-
  Use when building or modifying a Node.js / TypeScript app that uses the agent-squad npm package —
  multi-agent orchestration: orchestrator, agents (all built-in types + GroundedAgent), classifier
  routing (Bedrock / Anthropic / OpenAI), storage (in-memory / DynamoDB / SQL), retrievers
  (Amazon KB / Dakera), and tools (AgentTools + MCPToolProvider).
---

# agent-squad TypeScript — assistant guide

Node.js / TypeScript multi-agent orchestration framework (npm package `agent-squad`). All public
symbols are exported from a single barrel `typescript/src/index.ts`. This file is guidance and a
map — **not an API reference**. Read exact signatures from
`typescript/src/` and worked recipes from `docs/src/content/docs/`; this file tells you *what to
use, when, and what to watch out for*.

## When to use what

- **One assistant** → a single `Agent` subclass + `AgentSquad` with no routing. Or skip the
  orchestrator entirely and call `agent.processRequest(...)` directly.
- **Several specialists** → multiple agents registered with `orchestrator.addAgent(agent)`, a
  classifier routes each turn.
- **Answers must not drift from data** (prices, balances, live lookups) → `GroundedAgent`: a
  gatherer LLM calls tools, an isolated presenter LLM speaks only from the curated results.
- **Fixed pipeline** → `ChainAgent`: each agent's output is the next agent's input.
- **One lead LLM coordinating a team** → `SupervisorAgent`: the lead calls sub-agents as tools.
- **External tools via MCP** → `MCPToolProvider` (async factory pattern, optional peer dep).
- **RAG context** → attach a `Retriever` to any agent that supports `retriever?` in its options.

## How to install

```bash
npm install agent-squad
```

Optional peer dependencies — install only what you use:

| Package | Used by |
|---|---|
| `@aws-sdk/client-bedrock-runtime` | `BedrockLLMAgent`, `BedrockClassifier` (already a hard dep in current releases) |
| `@anthropic-ai/sdk` | `AnthropicAgent`, `AnthropicClassifier` (already a hard dep) |
| `openai` | `OpenAIAgent`, `OpenAIClassifier` (already a hard dep) |
| `@modelcontextprotocol/sdk` | `MCPToolProvider` — lazy `await import()` at connect time |
| `@dakera-ai/dakera` | `DakeraRetriever` — lazy `require()` at construction time |

`@modelcontextprotocol/sdk` and `@dakera-ai/dakera` are the only two true optional peer deps;
everything else ships as a hard dependency at the moment.

## How a turn works

`routeRequest` is the single entry point. It classifies the input, dispatches to the selected
agent, saves the exchange, and returns an `AgentResponse`. The response is either a plain string or
a Node.js `Transform` stream:

```typescript
import { AgentSquad, BedrockLLMAgent, BedrockClassifier } from 'agent-squad';

const orchestrator = new AgentSquad({
  classifier: new BedrockClassifier(),   // default when omitted
  // storage: new DynamoDbChatStorage(...),
  // config: { LOG_AGENT_CHAT: true, MAX_MESSAGE_PAIRS_PER_AGENT: 50 },
});

orchestrator.addAgent(new BedrockLLMAgent({
  name: 'Tech Support',
  description: 'Handles technical questions about software and hardware',
  streaming: true,
}));

const response = await orchestrator.routeRequest(
  userInput,
  userId,
  sessionId,
  additionalParams   // optional Record<string, any>
);

if (response.streaming) {
  // response.output is an AccumulatorTransform (Node.js Transform)
  for await (const chunk of response.output) {
    process.stdout.write(chunk);
  }
} else {
  // response.output is a string
  console.log(response.output);
  // response.thinking? is set when the agent used extended thinking
}

// response.metadata: { agentId, agentName, userId, sessionId, userInput, additionalParams }
```

`routeRequest` never throws — it catches all errors and returns them as a non-streaming
`AgentResponse` with the error string in `output` (configurable via `GENERAL_ROUTING_ERROR_MSG_MESSAGE`).

## The pieces

### Orchestrator: `AgentSquad`

```typescript
new AgentSquad(options?: OrchestratorOptions)
```

Key `OrchestratorOptions` fields:

| Field | Default | Notes |
|---|---|---|
| `classifier` | `new BedrockClassifier()` | Any `Classifier` subclass |
| `storage` | `new InMemoryChatStorage()` | Any `ChatStorage` subclass |
| `defaultAgent` | `undefined` | Used when classifier returns no match and `USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED` is true |
| `config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED` | `true` | Fall back to `defaultAgent` or return `NO_SELECTED_AGENT_MESSAGE` |
| `config.MAX_MESSAGE_PAIRS_PER_AGENT` | `100` | Per-agent history cap (pairs = user+assistant) |
| `config.MAX_RETRIES` | `3` | Classifier retries on bad XML response |
| `config.LOG_AGENT_CHAT` | `false` | |

Useful methods: `addAgent(agent)`, `setDefaultAgent(agent)`, `getDefaultAgent()`,
`getAllAgents()`, `analyzeAgentOverlap()`, `classifyRequest(...)`, `agentProcessRequest(...)`.

The classifier is exposed as a public field (`orchestrator.classifier`) so its system prompt can
be overridden after construction.

### Agents

All agents extend `Agent` and require at minimum `{ name, description }` in their options.

**`agent.id`** is derived automatically from `name`: non-alphanumeric stripped, spaces → hyphens,
lowercased. "Tech Support" → `"tech-support"`. This is the key used for storage and classifier
matching — it must be stable across restarts.

| Class | Options type | Notes |
|---|---|---|
| `BedrockLLMAgent` | `BedrockLLMAgentOptions` | Bedrock Converse API; supports `streaming`, `modelId`, `inferenceConfig`, `guardrailConfig`, `reasoningConfig`, `retriever`, `toolConfig`, `customSystemPrompt`, `client`, `callbacks` |
| `AnthropicAgent` | `AnthropicAgentOptions` | Direct Anthropic SDK; similar options shape |
| `OpenAIAgent` | `OpenAIAgentOptions` | OpenAI Chat Completions |
| `AmazonBedrockAgent` | `AmazonBedrockAgentOptions` | Amazon Bedrock Agents (pre-built agents, not Converse) |
| `BedrockInlineAgent` | `BedrockInlineAgentOptions` | Bedrock inline agents |
| `BedrockFlowsAgent` | `BedrockFlowsAgentOptions` | Bedrock Flows |
| `LambdaAgent` | `LambdaAgentOptions` | Invokes a Lambda function as an agent |
| `LexBotAgent` | `LexBotAgentOptions` | Amazon Lex V2 bot |
| `ChainAgent` | `ChainAgentOptions` | Fixed pipeline; `agents: Agent[]`, `defaultOutput?` |
| `SupervisorAgent` | `SupervisorAgentOptions` | Lead + team; `leadAgent` must be `BedrockLLMAgent` or `AnthropicAgent`; lead must have no `toolConfig` (SupervisorAgent manages tools) |
| `GroundedAgent` | `GroundedAgentOptions` | 2-LLM anti-hallucination; `gatherer`, `presenter`, `tools`, `curator?`, `presenterPrompt?` |

`AgentOptions` base fields: `name` (required), `description` (required), `saveChat?` (default
`true`), `logger?`, `LOG_AGENT_DEBUG_TRACE?`.

`BedrockLLMAgent` `toolConfig` shape:
```typescript
toolConfig: {
  tool: AgentTools | Tool[],   // AgentTools instance or raw Bedrock Tool array
  useToolHandler: (response: any, conversation: ConversationMessage[]) => any,
  toolMaxRecursions?: number,
}
```
When using `MCPToolProvider`, pass it as `toolConfig.tool` and omit `useToolHandler` — the
provider overrides `toolHandler` internally.

### GroundedAgent

Two-LLM anti-hallucination pattern. The gatherer calls tools; the presenter receives only the
curated facts (never raw tool output, never chat history from the gatherer):

```typescript
import {
  GroundedAgent, DataBlockCurator, PerToolCurator, PresenterPrompt,
  BedrockLLMAgent, AgentTools, AgentTool,
} from 'agent-squad';

const tools = new AgentTools([
  new AgentTool({ name: 'get_price', description: '...', func: async ({ sku }) => fetchPrice(sku) }),
]);

const gatherer = new BedrockLLMAgent({ name: 'Gatherer', description: '...', toolConfig: { tool: tools, useToolHandler: ... } });
const presenter = new BedrockLLMAgent({ name: 'Presenter', description: '...' });

const agent = new GroundedAgent({
  name: 'Price Agent',
  description: 'Answers pricing questions grounded in live data',
  gatherer,
  presenter,
  tools,
  curator: new DataBlockCurator(),          // default; or PerToolCurator for per-tool formatting
  presenterPrompt: PresenterPrompt.default(), // generic grounding prompt; or per-tool map
});
```

A no-tool turn (chit-chat) is answered by the gatherer directly, skipping the presenter.

### Classifiers

| Class | Options type | Notes |
|---|---|---|
| `BedrockClassifier` | `BedrockClassifierOptions` | Default when no classifier is passed to `AgentSquad` |
| `AnthropicClassifier` | `AnthropicClassifierOptions` | |
| `OpenAIClassifier` | `OpenAIClassifierOptions` | |

All classifiers support `setSystemPrompt(template?, variables?)` to override the routing prompt.
Template variables use `{{VAR_NAME}}` syntax; `AGENT_DESCRIPTIONS` and `HISTORY` are always
injected automatically.

### Storage

| Class | Notes |
|---|---|
| `InMemoryChatStorage` | Default; non-persistent; fine for dev and tests |
| `DynamoDbChatStorage` | Requires `@aws-sdk/client-dynamodb` and `@aws-sdk/lib-dynamodb` (hard deps) |
| `SqlChatStorage` | Requires `@libsql/client` (hard dep); works with Turso or local libsql |
| `SummarizingChatStorage` | Wraps any storage; compresses history via a user-supplied `ChatSummarizer` callable when `fetchChat` returns more than `triggerAt * 2` messages; cache-based save-back |

Storage is keyed by `(userId, sessionId, agentId)`. `fetchAllChats(userId, sessionId)` is used by
the classifier to get cross-agent history for context.

### Retrievers

| Class | Options type | Notes |
|---|---|---|
| `AmazonKnowledgeBasesRetriever` | `AmazonKnowledgeBasesRetrieverOptions` | Amazon Bedrock Knowledge Bases |
| `DakeraRetriever` | `DakeraRetrieverOptions` | Dakera memory server; optional peer dep `@dakera-ai/dakera` |

`DakeraRetrieverOptions`: `namespace` (required), `apiKey?` (falls back to `DAKERA_API_KEY` env),
`url?` (falls back to `DAKERA_URL` then `http://localhost:3000`), `topK?` (default 10), `filter?`.

Attach to a `BedrockLLMAgent` via `retriever:` option. The agent calls `retriever.retrieveAndCombineResults(inputText)` to augment its system prompt.

`DakeraRetriever.retrieveAndGenerate()` always throws — Dakera is retrieval-only.

### Tools: `AgentTools` and `AgentTool`

```typescript
import { AgentTools, AgentTool } from 'agent-squad';

const myTools = new AgentTools([
  new AgentTool({
    name: 'search_web',
    description: 'Search the web for current information',
    properties: {
      query: { type: 'string', description: 'The search query' },
    },
    required: ['query'],
    func: async ({ query }) => webSearch(query),
  }),
]);
```

`AgentTool` constructor will auto-extract parameter names from `func` if `properties` is omitted —
but this is fragile with minification. Always pass explicit `properties` and `required`.

### MCPToolProvider

`MCPToolProvider` extends `AgentTools`. Always use the async factory — never `new MCPToolProvider(...)` directly — so that tool definitions are fetched before the agent makes its first API call:

```typescript
import { MCPToolProvider } from 'agent-squad';

const provider = await MCPToolProvider.create([
  { type: 'stdio', command: 'uvx', args: ['my-mcp-server'] },
  { type: 'sse', url: 'http://localhost:3000/sse', headers: { Authorization: 'Bearer tok' } },
]);

const agent = new BedrockLLMAgent({
  name: 'MCP Agent',
  description: 'Agent with MCP tools',
  toolConfig: { tool: provider },
});

// Clean up when done (closes stdio processes and SSE connections)
await provider.disconnect();
```

`MCPServerConfig.type` is `"stdio"` or `"sse"`. For `stdio`: `command` is required, `args?` and
`env?` are optional. For `sse`: `url` is required, `headers?` is optional.

`MCPToolProvider` overrides `toBedrockFormat()`, `toAnthropicFormat()`, and `toOpenAIFormat()` to
pass MCP `inputSchema` through unchanged rather than re-serializing it.

Requires `npm install @modelcontextprotocol/sdk`. The SDK is imported lazily via `await import()`
inside `ensureConnected()` — installing agent-squad without the SDK is safe as long as you don't
instantiate `MCPToolProvider`.

## Custom implementations

Extend the abstract base class and pass your type where the built-in goes.

| Seam | Base class | Method to implement | Source |
|---|---|---|---|
| Agent | `Agent` | `processRequest(inputText, userId, sessionId, chatHistory, additionalParams?)` returns `Promise<ConversationMessage \| AsyncIterable<any>>` | `typescript/src/agents/agent.ts` |
| Classifier | `Classifier` | `processRequest(inputText, chatHistory)` returns `Promise<ClassifierResult>` | `typescript/src/classifiers/classifier.ts` |
| Storage | `ChatStorage` | `saveChatMessage`, `fetchChat`, `fetchAllChats` | `typescript/src/storage/chatStorage.ts` |
| Retriever | `Retriever` | `retrieve`, `retrieveAndCombineResults`, `retrieveAndGenerate` | `typescript/src/retrievers/retriever.ts` |

`ClassifierResult` shape: `{ selectedAgent: Agent | null, confidence: number }`.

`Classifier` base class provides `setAgents`, `setHistory`, `setSystemPrompt`, and
`getAgentById(agentId)` — use `getAgentById` in your `processRequest` to look up the selected agent
from the classifier's registered map.

## Gotchas

- **`agentId` is derived from `name`** at construction time: non-alphanumeric stripped, spaces
  replaced with `-`, lowercased. Changing an agent's `name` changes its `id`, which breaks chat
  history lookups in storage. Keep names stable across deployments.

- **Streaming response is a Node.js Transform stream**, not an async generator. Check
  `response.streaming` before iterating. The transform accumulates the full response internally;
  `for await (const chunk of response.output)` works because `Transform` implements
  `AsyncIterable`. Do not call `response.output.read()` manually.

- **`routeRequest` never throws**. Errors are returned as non-streaming `AgentResponse` with the
  error string in `output`. If you need to distinguish errors from real responses, check
  `response.metadata.errorType === 'classification_failed'` or inspect `metadata.agentId`.

- **`MCPToolProvider.create(...)` must be awaited before the agent is used**. The constructor alone
  does not connect; calling `processRequest` before `create` resolves means tool definitions are
  empty and the agent will behave as if it has no tools.

- **`BedrockClassifier` is the default**. If boto3/AWS credentials are not configured and you
  don't pass an explicit `classifier`, `AgentSquad` will construct a `BedrockClassifier` that will
  fail at runtime. Pass `classifier: new AnthropicClassifier(...)` or `new OpenAIClassifier(...)`
  if you're not on AWS.

- **Optional peer deps use lazy import/require**. `MCPToolProvider` uses `await import(...)` inside
  `ensureConnected()`; `DakeraRetriever` uses `require(...)` inside the constructor. Neither adds a
  top-level import, so a missing peer dep is only discovered at instantiation time — not at module
  load. Throw the missing-dep error early, before user input arrives.

- **`SupervisorAgent` restrictions**: `leadAgent` must be `BedrockLLMAgent` or `AnthropicAgent`;
  the lead agent must have no `toolConfig` set (SupervisorAgent wires its own tool loop). Pass
  additional native tools via `extraTools`.

- **`saveChat` defaults to `true`**. Every agent persists both sides of each exchange after the
  turn completes. Set `saveChat: false` on agents that should not write to storage (e.g. a
  presenter inside a `GroundedAgent` that is managed externally).

- **`additionalParams`** flows through `routeRequest` → `dispatchToAgent` → `agent.processRequest`.
  Use it to pass per-request context (tenant ID, request ID, feature flags) without touching agent
  options. The values end up in `response.metadata.additionalParams`.

- **`AgentTools` auto-extracts parameter names from `func` via `.toString()`**. This breaks with
  minification and TypeScript arrow functions with destructured arguments. Always supply explicit
  `properties` and `required` arrays to `AgentTool`.

- **`ThinkingResponse`**: when a `BedrockLLMAgent` is configured with `reasoningConfig`, the
  non-streaming path may return `response.thinking` (a string) alongside `response.output`. The
  streaming path does not surface thinking tokens separately.

## Go deeper

- **Prose & recipes** — `docs/src/content/docs/` (run the site from `docs/` with `npm run dev`):
  `orchestrator/overview`, `agents/built-in/bedrock-llm-agent`, `agents/built-in/grounded-agent`,
  `classifiers/overview`, `storage/overview`, `retrievers/overview`, `tools/mcp`.
- **Exact signatures** — `typescript/src/` (`orchestrator.ts`, `agents/`, `classifiers/`,
  `storage/`, `retrievers/`, `tools/mcpToolProvider.ts`, `utils/tool.ts`, `types/index.ts`).
- **Tests** — `typescript/tests/` for usage patterns and mock strategies (virtual mocks for
  optional peer deps via `jest.mock(..., { virtual: true })`).
- **Barrel** — `typescript/src/index.ts` is the definitive list of every public symbol.
