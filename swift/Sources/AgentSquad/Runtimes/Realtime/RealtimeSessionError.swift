/// Why a realtime turn or session ended abnormally — recorded on its trace spans (the tracer
/// stringifies it via `String(reflecting:)`, hence the debug description), never thrown to the
/// app: failures reach the app in-band as `RealtimeEvent.error`.
enum RealtimeSessionError: Error, CustomDebugStringConvertible {
    /// The transport's inbound stream ended without `stop()` — offline, server drop, or an
    /// asynchronously-rejected handshake. `underlying` is the transport's last receive error, if kept.
    case transportClosed(underlying: String?)
    /// A server `error` event ended the turn.
    case serverError(code: String?, message: String)
    /// A `response.done` arrived with `status: "failed"` — `detail` is its `status_details.error`.
    case responseFailed(detail: String?)
    /// The tool call reported a failure (`ToolResult.isError`) — fed back to the model as data,
    /// but a failure on the span all the same.
    case toolFailed(String)

    var debugDescription: String {
        switch self {
        case .transportClosed(let underlying):
            return "transport closed" + (underlying.map { ": \($0)" } ?? "")
        case .serverError(let code, let message):
            return "server error" + (code.map { " [\($0)]" } ?? "") + ": \(message)"
        case .responseFailed(let detail):
            return "response failed" + (detail.map { ": \($0)" } ?? "")
        case .toolFailed(let message):
            return "tool failed: \(message)"
        }
    }
}
