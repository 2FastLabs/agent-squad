import SwiftUI
import AgentSquad

struct ChatItem: Identifiable {
    let id = UUID()
    var text: String = ""
    var widget: UIPayload? = nil
    let isUser: Bool
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var items: [ChatItem] = []

    /// Header toggle — flips between the two modes from the article.
    @Published var showWidgets = true

    private let withWidgets: Orchestrator    // GroundedAgent with ui: .forward
    private let textOnly: Orchestrator       // GroundedAgent with ui: .suppress
    private let shopProvider = ShopToolProvider()

    init() {
        let key       = Config.openAIKey
        let brain     = ChatCompletionsClient(model: "gpt-4o", apiKey: key)       // fetches
        let presenter = ChatCompletionsClient(model: "gpt-4o-mini", apiKey: key)  // talks
        let provider  = shopProvider

        let gathererPrompt = """
            You are the data brain of a shopping assistant.
            GATHER the facts needed to answer the user — never write the final reply.

            Tools:
              get_order(orderId) → order status + delivery estimate

            Rules:
            - Call whatever tools you need to answer the question.
            - Use the chat history to resolve follow-ups.
            - Never invent values. If a tool returns nothing, say so.
            - Do NOT address the user or format anything — the presenter does that.
            """

        let presenterPrompt = PresenterPrompt(
            default: "Present the data to the user. Use ONLY what's provided. Be concise and natural.",
            perTool: [
                "get_order": """
                    Present an order status. State the order ID, current status, and estimated
                    delivery in one sentence. Use only the data provided.
                    """
            ]
        )

        func makeAgent(ui: UIPolicy) -> GroundedAgent {
            GroundedAgent(
                name: "Shop",
                description: "Product & order help, grounded in real data.",
                gatherer: brain,
                presenter: presenter,
                tools: provider,
                curator: .dataBlock,
                gathererPrompt: gathererPrompt,
                presenterPrompt: presenterPrompt,
                ui: ui
            )
        }

        // DeviceChatStorage is SwiftData (iOS 17+). Fall back to in-memory if it can't open a store
        // (disk pressure, sandbox) — or on iOS 16, where DeviceChatStorage is unavailable.
        let store: any ChatStorage = (try? DeviceChatStorage(userId: "u1")) ?? InMemoryChatStorage()
        withWidgets = Orchestrator(agents: [makeAgent(ui: .forward)],  store: store)
        textOnly    = Orchestrator(agents: [makeAgent(ui: .suppress)], store: store)
    }

    func send(_ text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        items.append(ChatItem(text: clean, isUser: true))
        items.append(ChatItem(isUser: false))           // assistant reply, filled in as events arrive
        let index = items.count - 1
        let router = showWidgets ? withWidgets : textOnly

        Task {
            do {
                for try await event in router.route(.text(clean), userId: "u1", sessionId: "s1") {
                    switch event {
                    case .textDelta(let token): items[index].text += token
                    case .widget(let payload):  items[index].widget = payload
                    case .error(let message):   items[index].text += "\n⚠️ \(message)"
                    default: break
                    }
                }
            } catch {
                items[index].text += "\n⚠️ \(error.localizedDescription)"
            }
        }
    }

    /// Called when a rendered widget invokes an `.app`-only tool (e.g. the Refresh button).
    /// We call the provider directly — the model is never involved — and push the fresh `UIPayload`
    /// back into the same chat item, which re-hydrates the widget in place.
    func handleWidgetTool(_ name: String, arguments: JSONValue, for itemID: UUID) {
        Task {
            guard let index = items.firstIndex(where: { $0.id == itemID }) else { return }
            if let payload = try? await shopProvider.call(name, arguments: arguments).ui {
                items[index].widget = payload
            }
        }
    }
}
