---
title: SummarizingChatStorage
description: Wrap any ChatStorage with an LLM summarizer to keep agent context small as conversations grow.
---

`SummarizingChatStorage` wraps any `ChatStorage` and automatically compresses long conversation histories. When a conversation exceeds a configurable size, a user-supplied async **summarizer** is called to condense the older messages while keeping a fixed number of recent pairs verbatim. The compressed result is held in an in-memory buffer — the inner store is never modified by the summarizer.

```swift
import AgentSquad

let storage = SummarizingChatStorage(
    wrapping: FileChatStorage(),
    summarizer: { history, keepLast in
        let old = Array(history.dropLast(keepLast * 2))
        let recent = Array(history.suffix(keepLast * 2))
        let summaryText = try await myLLM.summarize(old)
        let summary = ConversationMessage(role: .user, text: "[Summary]: \(summaryText)")
        return [summary] + recent
    },
    triggerAt: 20,  // compress when history exceeds 20 pairs (40 messages)
    keepLast: 2     // keep the 2 most recent pairs verbatim
)
let orchestrator = Orchestrator(agents: [agent], store: storage)
```

---

## How it works

The wrapper uses a **lazy in-memory buffer** per `(userId, sessionId, agentId)` slot:

1. **Before activation** — reads and writes are pure delegations to the inner store.
2. **Activation** — on the first `fetch` call whose raw history exceeds `triggerAt` message pairs (strictly `> triggerAt * 2` messages), the summarizer is called. The compressed result is stored in the buffer. The inner store is not written.
3. **Once active** — every `save` / `saveMessages` appends the new message(s) to the buffer and, if the buffer now exceeds the threshold again, calls the summarizer immediately. Fetches always read from the buffer — no LLM call on read.
4. **`fetchAllChats` is never intercepted** — the raw, full history from the inner store is always returned as-is. This is what the Classifier uses for cross-agent routing; it always sees the real conversation.

This pattern is inspired by LangChain's `ConversationSummaryBufferMemory`: compress eagerly on save, not lazily on fetch.

---

## The `ChatSummarizer` type

```swift
public typealias ChatSummarizer = @Sendable (
    _ history: [ConversationMessage],
    _ keepLast: Int
) async throws -> [ConversationMessage]
```

Your summarizer receives:

- `history` — the current buffer (everything since activation, including any already-compressed prefix from a previous round of summarization)
- `keepLast` — the configured `keepLast` value

It must return the new buffer — typically a summary message followed by the last `keepLast` pairs:

```swift
let mySummarizer: ChatSummarizer = { history, keepLast in
    let old = Array(history.dropLast(keepLast * 2))
    let recent = Array(history.suffix(keepLast * 2))
    let summaryText = try await myLLM.summarize(old)
    let summary = ConversationMessage(role: .user, text: "[Summary]: \(summaryText)")
    return [summary] + recent
}
```

The summarizer is `async throws` — it can call a remote LLM, a local model, or any async operation. If it throws, the error propagates from whichever `save` or `fetch` triggered it.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `wrapping` | — | The inner `ChatStorage` to wrap. |
| `summarizer` | — | The `ChatSummarizer` called to compress the buffer. |
| `triggerAt` | `20` | Number of message **pairs** above which the buffer is compressed. A buffer of 42 messages with `triggerAt: 20` triggers because `42 > 40`. |
| `keepLast` | `2` | Number of most-recent message pairs to keep verbatim. Passed to the summarizer as `keepLast`. |

---

## Key properties

**Raw history is never modified.** The inner store receives every raw message. The summarizer only affects the in-memory buffer.

**Fetch is always fast.** Summarization runs during `save`, so `fetch` just reads the buffer — no LLM call.

**`fetchAllChats` bypasses the buffer.** The Classifier always routes from the real, unsummarized cross-agent history.

**Buffer state is in-memory.** If the process restarts, the buffer is cold. The first qualifying `fetch` after restart will re-activate the buffer from the inner store and call the summarizer once.

---

## Composing with TransformingChatStorage

Both wrappers implement `ChatStorage` so they compose freely. A typical production stack scrubs PII before storage and then summarizes to keep context small:

```swift
// Inner: SwiftData store
// Middle: scrub PII before anything reaches disk
// Outer: summarize to keep agent context small
let store = SummarizingChatStorage(
    wrapping: TransformingChatStorage(
        wrapping: try DeviceChatStorage(userId: userId),
        transform: scrubPIITransform
    ),
    summarizer: mySummarizer,
    triggerAt: 20,
    keepLast: 2
)
```

The order matters: `TransformingChatStorage` scrubs raw messages before they hit disk; `SummarizingChatStorage` operates on the scrubbed versions when it builds its in-memory buffer.

---

## Related pages

- [Storage overview](/agent-squad/swift/storage/overview/) — the `ChatStorage` protocol
- [TransformingChatStorage](/agent-squad/swift/storage/built-in/transforming/) — scrub PII / drop messages before persistence
- [File](/agent-squad/swift/storage/built-in/file/) · [Device (SwiftData)](/agent-squad/swift/storage/built-in/device/) · [In-memory](/agent-squad/swift/storage/built-in/in-memory/) — stores to wrap
- [Custom store](/agent-squad/swift/storage/custom/) — implementing `ChatStorage` yourself
