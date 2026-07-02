import Foundation

/// What the isolated presenter sees besides its prompt. It never sees chat history, tools, or the
/// gatherer's transcript in either mode.
public enum PresenterInput: Sendable {
    /// This turn's question plus the curated data (the default) — the presenter answers the user
    /// directly: it can select, frame a yes/no, and match the user's language.
    case questionAndData
    /// Only the curated data — a pure explainer. Any answer-framing must already be in the data.
    case dataOnly
}

/// Chooses the presenter's system prompt, keyed by the turn's primary tool. One prompt by default;
/// supply a per-tool map for tool-specific presenters.
public struct PresenterPrompt: Sendable {
    private let defaultPrompt: String
    private let perTool: [String: String]

    public init(default defaultPrompt: String, perTool: [String: String] = [:]) {
        self.defaultPrompt = defaultPrompt
        self.perTool = perTool
    }

    /// The prompt for a turn whose primary tool is `primaryTool` (falls back to the default).
    public func resolve(primaryTool: String?) -> String {
        if let primaryTool, let prompt = perTool[primaryTool] { return prompt }
        return defaultPrompt
    }

    /// A generic grounding instruction — present only the provided data, never invent values.
    public static let `default` = PresenterPrompt(default: """
        You are presenting information to the user. Use ONLY the data provided. Be concise and \
        natural, and never invent or infer values that are not present in the data. If the \
        question asserts something the data contradicts, trust the data.
        """)
}
