import SwiftUI
import AgentSquad

/// The native SwiftUI widget for `ui://shop/order-card`. It's hydrated from the tool's render-only
/// `structuredContent` — no HTML, no web view. This is your own view; style it however you like.
struct OrderCardView: View {
    let data: JSONValue
    let onRefresh: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Order \(string("orderId"))").font(.headline)
            Text(string("status")).font(.subheadline).bold().foregroundStyle(.green)
            HStack(spacing: 4) {
                Text("ETA").foregroundStyle(.secondary)
                Text(string("eta")).bold()
                Text("·").foregroundStyle(.secondary)
                Text(string("carrier"))
            }
            .font(.subheadline)

            // Arrays hydrate from structuredContent too.
            if case .array(let items)? = data["items"], !items.isEmpty {
                Divider().padding(.vertical, 2)
                ForEach(items.indices, id: \.self) { i in
                    if case .string(let name) = items[i] {
                        Text("• \(name)").font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }

            Button("Refresh", action: onRefresh)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color(uiColor: .secondarySystemBackground)))
    }

    /// Pull a string field out of the render-only data.
    private func string(_ key: String) -> String {
        if case .string(let value)? = data[key] { return value }
        return ""
    }
}
