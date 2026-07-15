import Foundation

enum Config {
    /// Set `OPENAI_API_KEY` in your scheme's environment variables, or paste a key below for a quick
    /// local test. Do NOT commit a real key.
    static let openAIKey: String = {
        if let env = ProcessInfo.processInfo.environment["OPENAI_API_KEY"], !env.isEmpty {
            return env
        }
        return "PASTE_YOUR_OPENAI_KEY_HERE"
    }()

    /// When set, tools come from a remote MCP server (streamable HTTP) instead of the in-app
    /// `ShopToolProvider` — e.g. the sample at `examples/mcp-shop-server`. Nil = native provider.
    /// The Simulator reaches your Mac at `127.0.0.1`; for a physical device, start the server with
    /// `HOST=0.0.0.0` and use your Mac's LAN IP here.
    static let mcpServerURL: String? = nil   // "http://127.0.0.1:8000/mcp"
}
