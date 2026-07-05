---
title: Storage overview
description: The ChatStorage protocol, conversation scoping, and how to choose between the three built-in stores.
---

AgentSquad separates memory from routing. Each agent receives its own scoped history; the [Orchestrator](/agent-squad/swift/orchestrator/overview/) also reads a merged cross-agent view so the Classifier can see the full conversation when selecting the next agent.

## The `ChatStorage` protocol

```swift
public protocol ChatStorage: Sendable {
    func fetch(
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws -> [ConversationMessage]

    func save(
        _ message: ConversationMessage,
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws

    func saveMessages(
        _ messages: [ConversationMessage],
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws

    /// Merged, timestamp-ordered history across all agents; assistant messages `[agentId]`-prefixed.
    func fetchAllChats(userId: String, sessionId: String) async throws -> [ConversationMessage]
}
```

### Conversation scope

`fetch` / `save` / `saveMessages` are keyed by a `(userId, sessionId, agentId)` triple and feed the selected agent. `fetchAllChats` returns a merged, timestamp-ordered view across **all** agents for the session; assistant messages are prefixed with `[agentId]` so the Classifier can attribute them.

See [Messages & events](/agent-squad/swift/reference/messages-and-events/) for the `ConversationMessage` type.

### `maxMessages` window

```swift
public enum ChatStorageDefaults {
    public static let maxMessages = 100   // counts messages, not pairs
}
```

`maxMessages` counts individual messages, not user/assistant pairs. The framework rounds any budget down to an even number so a user/assistant pair is never split. Pass `nil` to keep an unbounded history.

### Default implementations on `ChatStorage`

Two helpers are provided as protocol extensions — call them in your own store, do not reimplement them:

- **`trimToEvenPairs(_:maxMessages:)`** — trims to the most recent `maxMessages`, rounding down to even.
- **`isConsecutiveSameRole(_:_:)`** — returns `true` when a new message repeats the last stored role; stores drop such saves.

### `store: nil`

Passing `store: nil` to the Orchestrator or voice assistant disables persistence entirely — every session starts fresh. Useful during development or for ephemeral flows.

---

## Choosing a store

| Store | Platform | Persistence | Multi-agent `[agentId]` attribution |
|---|---|---|---|
| [`InMemoryChatStorage`](/agent-squad/swift/storage/built-in/in-memory/) | iOS 16+ | Session only | No |
| [`FileChatStorage`](/agent-squad/swift/storage/built-in/file/) | iOS 16+ | Disk (JSON) | Yes |
| [`DeviceChatStorage`](/agent-squad/swift/storage/built-in/device/) | iOS 17+ / macOS 14+ | Disk (SwiftData) | Yes |
| `store: nil` | Any | None | — |

For [voice](/agent-squad/swift/voice/overview/) sessions the same stores apply — pass the chosen instance to the voice assistant's `store:` parameter exactly as you would for a text Orchestrator.

Any of them can be wrapped in [`TransformingChatStorage`](/agent-squad/swift/storage/built-in/transforming/) to scrub PII or reshape messages before they are persisted.

:::note
`InMemoryChatStorage` is scope-agnostic and returns no `[agentId]` prefix from `fetchAllChats`. For multi-agent Orchestrators where the Classifier needs routing attribution, use one of the persistent stores.
:::

---

## Built-in stores

- [InMemoryChatStorage](/agent-squad/swift/storage/built-in/in-memory/) — non-persistent, seedable, iOS 16+
- [FileChatStorage](/agent-squad/swift/storage/built-in/file/) — JSON files, iOS 16+
- [DeviceChatStorage](/agent-squad/swift/storage/built-in/device/) — SwiftData, iOS 17+ / macOS 14+
- [TransformingChatStorage](/agent-squad/swift/storage/built-in/transforming/) — wraps any store; PII scrub / message transform before saving, iOS 16+

## Custom store

Need a remote database, an encrypted keychain bucket, or a shared app-group container? [Write a custom store](/agent-squad/swift/storage/custom/).
