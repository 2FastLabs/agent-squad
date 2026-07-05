---
title: TransformingChatStorage
description: Wrap any ChatStorage with a message transform that runs before persistence — strip PII, redact tokens, clip payloads, or drop messages entirely.
---

`TransformingChatStorage` wraps any `ChatStorage` — built-in or custom — and runs a `MessageTransform` on every message **before it is saved**. It is the seam for PII scrubbing and message shaping at the persistence boundary: what reaches disk is the transformed form, so scrubbed data also never re-enters prompts when history is fetched later.

```swift
import AgentSquad

public typealias MessageTransform = @Sendable (ConversationMessage) async throws -> ConversationMessage?

let store = TransformingChatStorage(wrapping: FileChatStorage()) { message in
    message.mappingText { text in
        text.replacing(#/\b[A-Z]{2}\d{2}(?: ?\d{4}){4,7}\b/#, with: "[IBAN]")
    }
}
let orchestrator = Orchestrator(agents: [agent], store: store)
```

Reads (`fetch`, `fetchAllChats`) pass through to the wrapped store untouched.

---

## The transform

| Return | Effect |
|---|---|
| a modified message | the modified form is persisted (same `id`/`role`/`timestamp` — see `mappingText`) |
| the message unchanged | persisted as-is |
| `nil` | the message is **dropped** — nothing persisted |
| `throw` | the save **fails loudly** — nothing is persisted unscrubbed |

The closure is `async`, so an on-device model (e.g. `NLTagger`-based entity detection) can do the scrubbing.

### `mappingText` — the common case

Most transforms only touch text. `ConversationMessage.mappingText(_:)` runs a closure over every string part — `.text` **and** `.audioTranscript` — and leaves structured parts (`toolCall`/`toolResult` payloads, widgets) alone, preserving `id`, `role`, and `timestamp`:

```swift
message.mappingText { $0.replacingOccurrences(of: cardNumber, with: "[CARD]") }
```

---

## Worked transforms

Reusable functions typed as `MessageTransform` — define once, pass to any wrapper.

**Pattern-based PII scrub** (IBANs, card-length digit runs, emails):

```swift
let scrubPII: MessageTransform = { message in
    message.mappingText { text in
        var scrubbed = text
        scrubbed = scrubbed.replacing(#/\b[A-Z]{2}\d{2}(?: ?\d{4}){4,7}\b/#, with: "[IBAN]")
        scrubbed = scrubbed.replacing(#/\b(?:\d[ -]?){13,19}\b/#, with: "[CARD]")
        scrubbed = scrubbed.replacing(#/[\w.+-]+@[\w-]+\.[\w.]+/#, with: "[EMAIL]")
        return scrubbed
    }
}

let store = TransformingChatStorage(wrapping: FileChatStorage(), transform: scrubPII)
```

**Name detection with an on-device model** (`NLTagger`) — collect ranges first, replace in reverse so earlier ranges stay valid:

```swift
import NaturalLanguage

let scrubNames: MessageTransform = { message in
    message.mappingText { text in
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text
        var ranges: [Range<String.Index>] = []
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType,
                             options: [.omitWhitespace, .omitPunctuation]) { tag, range in
            if tag == .personalName { ranges.append(range) }
            return true
        }
        var scrubbed = text
        for range in ranges.reversed() { scrubbed.replaceSubrange(range, with: "[NAME]") }
        return scrubbed
    }
}
```

**Dropping messages** — keep operator/debug turns out of history entirely (mind the pairing caution below):

```swift
let dropDebugTurns: MessageTransform = { message in
    message.text.hasPrefix("/debug") ? nil : message
}
```

**Failing closed** — when the scrubber itself can error, throw rather than persist raw data:

```swift
struct ScrubberUnavailable: Error {}

let strictScrub: MessageTransform = { message in
    // Scrubber = your on-device scrubbing service; stands in for whatever does the real work.
    guard await Scrubber.shared.isReady else { throw ScrubberUnavailable() }
    return await Scrubber.shared.scrub(message)
}
```

:::caution
Prefer **redacting** over dropping. Stores skip a save whose role repeats the last stored message's (the consecutive-same-role guard) — so dropping one side of an exchange can make the store silently skip its counterpart too, e.g. drop the assistant reply and the *next user message* is refused as user-after-user.
:::

---

## Composing

It's a plain `ChatStorage`, so it drops in anywhere one goes and wraps anything, including another wrapper:

```swift
let clipLongMessages: MessageTransform = { message in
    message.mappingText { String($0.prefix(4_000)) }
}

// Scrub FIRST, then clip, over the SwiftData store (the outer transform runs first).
// Scrub-before-clip matters: clipping mid-match (e.g. half a card number) would leave a
// fragment the scrub patterns no longer recognize.
let store = TransformingChatStorage(
    wrapping: TransformingChatStorage(wrapping: try DeviceChatStorage(userId: "u1"), transform: clipLongMessages),
    transform: scrubPII
)
```

:::note
This transforms what reaches **storage**. For scrubbing what reaches **trace backends**, supply a custom `Redactor` to the tracing pipeline instead — the two seams are independent.
:::

---

## Related pages

- [Storage overview](/agent-squad/swift/storage/overview/) — the `ChatStorage` protocol
- [File](/agent-squad/swift/storage/built-in/file/) · [Device (SwiftData)](/agent-squad/swift/storage/built-in/device/) · [In-memory](/agent-squad/swift/storage/built-in/in-memory/) — stores to wrap
- [Custom store](/agent-squad/swift/storage/custom/) — implementing `ChatStorage` yourself
