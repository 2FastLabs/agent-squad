/// One captured tool call, with its full `ToolResult` so the UI payload survives for the widget.
struct CapturedCall: Sendable {
    let name: String
    let result: ToolResult

    /// The narrow curator view: keeps the UI's resource URI as a presence signal but drops the full
    /// payload (the widget is sourced from `result.ui`). Shared with the Realtime runtime.
    var curatorView: CapturedTool {
        CapturedTool(name: name, ui: result.ui?.resourceURI, structuredContent: result.structuredContent, content: result.content)
    }
}

/// Shared grounding helpers for the gather → present pattern, used by `GroundedAgent` and the
/// Realtime runtime.
enum Grounding {
    /// The turn's primary tool: the last call advertising a UI, else the last call (so prompt
    /// selection still works without UI).
    static func primary(of calls: [CapturedCall]) -> CapturedCall? {
        calls.last(where: { $0.result.ui != nil }) ?? calls.last
    }

    /// The presenter's user message: curated data, tagged — preceded by the question when the
    /// `PresenterInput` mode includes it, so the model can tell the two apart.
    static func presenterMessage(question: String, data: String, mode: PresenterInput) -> String {
        let dataBlock = """
        <data to present>
        \(data)
        </data to present>
        """
        guard case .questionAndData = mode else { return dataBlock }
        return """
        <user question>
        \(question)
        </user question>
        \(dataBlock)
        """
    }
}
