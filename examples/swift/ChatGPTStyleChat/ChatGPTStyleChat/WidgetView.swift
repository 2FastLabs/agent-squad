import SwiftUI
import AgentSquad

/// Renders a tool-advertised widget as a **native SwiftUI view**, chosen by the payload's
/// `resourceURI` and hydrated from its render-only `structuredContent`. Add one `case` per widget
/// your tools return. (An MCP server's own HTML widget would instead be rendered in a `WKWebView`;
/// here everything is local Swift.)
struct WidgetView: View {
    let payload: UIPayload
    let onAppTool: (String, JSONValue) -> Void   // widget → app (e.g. the .app-only Refresh)

    var body: some View {
        switch payload.resourceURI {
        case ShopToolProvider.orderCardURI:
            OrderCardView(data: payload.structuredContent) {
                let orderId = payload.structuredContent["orderId"] ?? .string("")
                onAppTool("refresh_order", ["orderId": orderId])
            }
        default:
            EmptyView()
        }
    }
}
