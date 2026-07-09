---
title: Custom transport
description: Implement RealtimeTransport to swap in a different WebSocket library, add proxy headers, or inject a mock for unit tests.
---

`RealtimeTransport` is the seam between the session's control logic and the wire. Swap it to wrap a third-party WebSocket library, add runtime proxy headers, or inject a mock for unit tests without a live connection. Both `OpenAIVoiceAssistant` and `OpenAIGroundedVoiceAssistant` accept `transport: any RealtimeTransport`.

For the built-in `URLSessionWebSocketTransport` that covers most production use cases see [WebSocket transport](/agent-squad/swift/voice/built-in/websocket-transport/).

---

## Protocol requirements

```swift
public protocol RealtimeTransport: Sendable {
    func connect() async throws           // open the connection
    func send(_ json: String) async throws // send one JSON frame
    var events: AsyncStream<String> { get }// inbound frames; finishes on close
    func close() async                    // close the connection and finish events
    func lastReceiveError() async -> (any Error)?   // error that ended events (defaulted to nil)
}
```

:::note
`events` must be a `nonisolated let` so the protocol's synchronous getter is satisfied from outside the actor. Store the `AsyncStream.Continuation` as a matching `nonisolated let` and drive it from inside the actor — exactly the pattern `URLSessionWebSocketTransport` uses.
:::

`lastReceiveError()` has a protocol-extension default of `nil`, so existing transports keep compiling without it. Implement it (keep the error that killed your receive loop and return it here) so the session can attribute an unexpected close: it reads the value after `events` finishes, ends its trace spans with that error, and emits `.error(code: "transport_closed", …)` with the underlying detail instead of a bare "closed".

---

## Example: mock transport for unit tests

An `actor` satisfies `Sendable` automatically and prevents data races on shared state. The pattern mirrors `URLSessionWebSocketTransport`: a stored `continuation` drives `events`, `connect`/`close` guard against double-open/double-close, and `send` buffers frames so tests can assert on them.

```swift
import AgentSquad

public actor MockRealtimeTransport: RealtimeTransport {
    public nonisolated let events: AsyncStream<String>
    private nonisolated let continuation: AsyncStream<String>.Continuation

    /// Frames sent by the session — inspect in assertions.
    public private(set) var sentFrames: [String] = []
    private var connected = false

    public init() {
        (self.events, self.continuation) = AsyncStream.makeStream(of: String.self)
    }

    public func connect() async throws {
        guard !connected else { throw RealtimeTransportError.alreadyConnected }
        connected = true
    }

    public func send(_ json: String) async throws {
        guard connected else { throw RealtimeTransportError.notConnected }
        sentFrames.append(json)
    }

    /// Simulate a server-sent frame (call from your test to drive the session).
    public func receive(_ json: String) {
        continuation.yield(json)
    }

    public func close() async {
        connected = false
        continuation.finish()
    }
}
```

---

## Injecting into a voice assistant

Both `OpenAIVoiceAssistant` and `OpenAIGroundedVoiceAssistant` accept `transport: any RealtimeTransport`, so substitution is a single call-site change:

```swift
// Production
let transport = URLSessionWebSocketTransport(apiKey: "sk-...")

// Tests / custom backend
let transport = MockRealtimeTransport()

let assistant = OpenAIVoiceAssistant(
    name: "voice-assistant",
    transport: transport,   // <-- any RealtimeTransport
    tools: myToolProvider,
    userId: "u1",
    sessionId: UUID().uuidString
)
```

---

## Example: custom backend with extra headers

For a non-OpenAI endpoint or a proxy that requires custom auth headers, implement `RealtimeTransport` directly rather than forking `URLSessionWebSocketTransport`.

```swift
import AgentSquad
import Foundation

public actor CustomWebSocketTransport: RealtimeTransport {
    public nonisolated let events: AsyncStream<String>
    private nonisolated let continuation: AsyncStream<String>.Continuation

    private let request: URLRequest
    private let session: URLSession
    private var task: URLSessionWebSocketTask?
    private var receiveLoop: Task<Void, Never>?
    private var receiveError: (any Error)?

    public init(url: URL, bearerToken: String, extraHeaders: [String: String] = [:]) {
        var req = URLRequest(url: url)
        req.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        for (field, value) in extraHeaders { req.setValue(value, forHTTPHeaderField: field) }
        self.request = req
        self.session = .shared
        (self.events, self.continuation) = AsyncStream.makeStream(of: String.self)
    }

    public func connect() async throws {
        guard task == nil else { throw RealtimeTransportError.alreadyConnected }
        let task = session.webSocketTask(with: request)
        self.task = task
        task.resume()
        receiveLoop = Task { [weak self] in await self?.pump() }
    }

    public func send(_ json: String) async throws {
        guard let task else { throw RealtimeTransportError.notConnected }
        try await task.send(.string(json))
    }

    public func close() async {
        receiveLoop?.cancel()
        receiveLoop = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        continuation.finish()
    }

    // Lets the session attribute an unexpected close (transport_closed) to the real cause.
    public func lastReceiveError() async -> (any Error)? { receiveError }

    private func pump() async {
        guard let task else { return }
        while !Task.isCancelled {
            do {
                switch try await task.receive() {
                case .string(let text):   continuation.yield(text)
                case .data(let data):     continuation.yield(String(decoding: data, as: UTF8.self))
                @unknown default:         break
                }
            } catch {
                receiveError = error   // keep for the session's post-mortem
                continuation.finish()
                return
            }
        }
    }
}
```

---

## Related pages

- [Voice overview](/agent-squad/swift/voice/overview/) — `RealtimeRuntime`, protocols, and event reference
- [WebSocket transport](/agent-squad/swift/voice/built-in/websocket-transport/) — `URLSessionWebSocketTransport` reference
- [OpenAIVoiceAssistant](/agent-squad/swift/voice/built-in/openai-voice/) — single-LLM built-in
- [OpenAIGroundedVoiceAssistant](/agent-squad/swift/voice/built-in/openai-grounded-voice/) — grounded two-phase built-in
