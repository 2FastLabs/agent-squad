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
}
