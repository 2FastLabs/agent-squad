import SwiftUI
import WebKit
import AgentSquad

/// Renders a `UIPayload` HTML template inside a locked-down `WKWebView`:
///  • injects `structuredContent` as `window.WIDGET_DATA` (render-only — the model never sees it)
///  • enforces the payload's `UISecurity` as a Content-Security-Policy
///  • bridges widget → app so `.app`-only tools (Refresh) can call back into Swift
struct WidgetHostView: UIViewRepresentable {
    let payload: UIPayload
    let onAppTool: @MainActor (String, JSONValue) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onAppTool: onAppTool) }

    func makeUIView(context: Context) -> WKWebView {
        let controller = WKUserContentController()
        controller.add(context.coordinator, name: "host")

        // Inject the render-only data BEFORE any page script runs.
        controller.addUserScript(
            WKUserScript(source: "window.WIDGET_DATA = \(jsonString(payload.structuredContent));",
                         injectionTime: .atDocumentStart,
                         forMainFrameOnly: true)
        )

        let config = WKWebViewConfiguration()
        config.userContentController = controller
        config.defaultWebpagePreferences.allowsContentJavaScript = true

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.scrollView.isScrollEnabled = false
        webView.isOpaque = false
        webView.backgroundColor = .clear

        if case .html(let html)? = payload.template {
            webView.loadHTMLString(cspMeta(payload.security) + html, baseURL: nil)
        }
        context.coordinator.lastData = jsonString(payload.structuredContent)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // If the payload's data changed (e.g. after Refresh), push it and re-render.
        let current = jsonString(payload.structuredContent)
        guard current != context.coordinator.lastData else { return }
        context.coordinator.lastData = current
        webView.evaluateJavaScript("window.WIDGET_DATA = \(current); window.render && window.render();")
    }

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "host")
    }

    // MARK: - Helpers

    private func jsonString(_ value: JSONValue) -> String {
        guard let data = try? JSONEncoder().encode(value),
              let string = String(data: data, encoding: .utf8) else { return "{}" }
        return string
    }

    /// Build a `<meta>` CSP from `UISecurity`. Undeclared domains are blocked.
    private func cspMeta(_ security: UISecurity?) -> String {
        let s = security ?? UISecurity()
        let connect = (["'self'"] + s.connectDomains).joined(separator: " ")
        let img     = (["'self'", "data:"] + s.resourceDomains).joined(separator: " ")
        let frame   = (["'none'"] + s.frameDomains).joined(separator: " ")
        return """
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; \
        style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src \(img); \
        connect-src \(connect); frame-src \(frame);">
        """
    }

    final class Coordinator: NSObject, WKScriptMessageHandler {
        let onAppTool: @MainActor (String, JSONValue) -> Void
        var lastData = ""

        init(onAppTool: @escaping @MainActor (String, JSONValue) -> Void) {
            self.onAppTool = onAppTool
        }

        func userContentController(_ uc: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            // WebKit delivers script messages on the main thread.
            guard let body = message.body as? String,
                  let data = body.data(using: .utf8),
                  let call = try? JSONDecoder().decode(AppToolCall.self, from: data) else { return }
            MainActor.assumeIsolated {
                onAppTool(call.tool, call.args ?? .object([:]))
            }
        }
    }

    private struct AppToolCall: Decodable {
        let tool: String
        let args: JSONValue?
    }
}
