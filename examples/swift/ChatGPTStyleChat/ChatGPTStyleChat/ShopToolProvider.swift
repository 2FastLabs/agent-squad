import AgentSquad
import Foundation

/// A native Swift tool provider — no backend required.
///
/// `get_order`     is callable by the model AND advertises a widget (`ui:`).
/// `refresh_order` is `.app`-only: only the rendered widget can call it, never the LLM.
struct ShopToolProvider: ToolProvider {

    static let orderCardURI = "ui://shop/order-card"

    /// Every tool this provider can run. `listTools()` advertises only the model-visible ones;
    /// `.app`-only tools stay callable via `call(_:arguments:)` but are never shown to the LLM.
    private let allTools: [AgentTool] = [
        AgentTool(
            name: "get_order",
            description: "Look up an order's status and delivery estimate.",
            inputSchema: [
                "type": "object",
                "properties": ["orderId": ["type": "string", "description": "The order ID"]],
                "required": ["orderId"]
            ],
            ui: Self.orderCardURI
        ),
        AgentTool(
            name: "refresh_order",
            description: "Refresh the order card with the latest status.",
            inputSchema: [
                "type": "object",
                "properties": ["orderId": ["type": "string"]],
                "required": ["orderId"]
            ],
            ui: Self.orderCardURI,
            visibility: .app          // widget-only; never offered to the model
        )
    ]

    /// Model-visible tools only — mirrors what the built-in `ToolKit` does internally. A custom
    /// provider must filter here itself, or an `.app` tool would still be advertised to the LLM.
    func listTools() async throws -> [AgentTool] {
        allTools.filter { $0.visibility.contains(.model) }
    }

    func call(_ name: String, arguments: JSONValue) async throws -> ToolResult {
        switch name {
        case "get_order", "refresh_order":
            guard case .string(let orderId)? = arguments["orderId"] else {
                return .failure("Missing required argument: orderId")
            }
            return order(orderId, refreshed: name == "refresh_order")
        default:
            return .failure("Unknown tool: \(name)")
        }
    }

    // MARK: - Faked backend (replace with your API / database)

    private func order(_ orderId: String, refreshed: Bool) -> ToolResult {
        let status  = refreshed ? "Out for delivery" : "In transit"
        let eta     = refreshed ? "Today, 6:00 PM"   : "2026-07-02"
        let carrier = "FastShip"

        // Single source of truth — both the text and the widget render from this.
        let data: JSONValue = [
            "orderId": .string(orderId),
            "status":  .string(status),
            "eta":     .string(eta),
            "carrier": .string(carrier),
            "items":   ["Wireless Headphones", "USB-C Cable"]
        ]

        return ToolResult(
            // text → the Brain's context (so it can reason / follow up)
            content: [.text("Order \(orderId): \(status), ETA \(eta) via \(carrier).")],
            // structured data → not added to the model's context; hydrates the curator + widget
            structuredContent: data,
            // the self-contained widget package
            ui: UIPayload(
                resourceURI: Self.orderCardURI,
                mimeType: "text/html;profile=mcp-app",
                template: .html(OrderCard.html),
                structuredContent: data,
                security: UISecurity(
                    resourceDomains: ["https://cdn.myshop.com"],  // images allowed from here
                    prefersBorder: true
                )
            )
        )
    }
}
