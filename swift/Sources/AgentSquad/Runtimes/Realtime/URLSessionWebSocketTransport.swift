import Foundation

/// The real `RealtimeTransport`: an OpenAI Realtime connection over `URLSessionWebSocketTask`.
/// URLSession-only (no dependency), so it lives in core. The session's control logic is unit-tested
/// against a mock transport; this is the thin live-socket adapter (connect, a receive loop that
/// yields text frames, send, close).
///
/// An `actor` so the task and receive loop can't race; `events` is a `nonisolated let` to satisfy the
/// synchronous protocol getter. Construct one per connection and `close()` when done.
public actor URLSessionWebSocketTransport: RealtimeTransport {
    public nonisolated let events: AsyncStream<String>
    private nonisolated let continuation: AsyncStream<String>.Continuation

    private let endpoint: URL
    private let headers: [String: String]
    private let session: URLSession
    private var task: URLSessionWebSocketTask?
    private var receiveLoop: Task<Void, Never>?
    private var receiveError: (any Error)?

    public init(
        url: URL = URL(string: "wss://api.openai.com/v1/realtime")!,
        model: String = "gpt-realtime",
        apiKey: String,
        headers: [String: String] = [:],
        session: URLSession = .shared
    ) {
        self.endpoint = Self.endpoint(url, model: model)
        var headers = headers
        headers["Authorization"] = "Bearer \(apiKey)"
        self.headers = headers
        self.session = session
        (self.events, self.continuation) = AsyncStream.makeStream(of: String.self)
    }

    public func connect() async throws {
        guard task == nil else { throw RealtimeTransportError.alreadyConnected }   // connect once per instance
        var request = URLRequest(url: endpoint)
        for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
        let task = session.webSocketTask(with: request)
        self.task = task
        task.resume()
        receiveLoop = Task { [weak self] in await self?.receive() }
    }

    public func send(_ json: String) async throws {
        guard let task else { throw RealtimeTransportError.notConnected }
        try await task.send(.string(json))
    }

    public func lastReceiveError() async -> (any Error)? { receiveError }

    public func close() async {
        receiveLoop?.cancel()
        receiveLoop = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        continuation.finish()
    }

    // MARK: - Internals

    /// Read frames until the socket errors or closes, then finish `events` (which ends the session's
    /// pump). `receive()` must be re-armed after each message — hence the loop.
    private func receive() async {
        guard let task else { return }
        while !Task.isCancelled {
            do {
                switch try await task.receive() {
                case .string(let text): continuation.yield(text)
                case .data(let data): continuation.yield(String(decoding: data, as: UTF8.self))
                @unknown default: break
                }
            } catch {
                receiveError = error   // kept for the session's post-mortem (`lastReceiveError`)
                continuation.finish()
                return
            }
        }
    }

    /// Append `?model=…` to the base realtime URL (the model is passed both here and in `session.update`).
    static func endpoint(_ base: URL, model: String) -> URL {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return base }
        components.queryItems = (components.queryItems ?? []) + [URLQueryItem(name: "model", value: model)]
        return components.url ?? base
    }
}
