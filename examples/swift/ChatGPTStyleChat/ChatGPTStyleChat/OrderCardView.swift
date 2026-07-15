import SwiftUI
import AgentSquad

struct OrderCardView: View {
    let data: JSONValue
    let onRefresh: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // ── Top: gradient header ──────────────────────────────────
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        // "WIDGET" label — makes it obvious this is a live card
                        Label("Live widget", systemImage: "square.grid.2x2.fill")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.7))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(.white.opacity(0.15), in: Capsule())

                        Text("Order \(string("orderId"))")
                            .font(.title3.weight(.bold))
                            .foregroundStyle(.white)

                        // Status badge
                        Text(string("status").capitalized)
                            .font(.caption.weight(.bold))
                            .foregroundStyle(statusColors.0)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(.white, in: Capsule())
                    }
                    Spacer()
                    Image(systemName: statusIcon)
                        .font(.system(size: 40))
                        .foregroundStyle(.white.opacity(0.25))
                }

                Divider().overlay(.white.opacity(0.25))

                // ETA row
                HStack(spacing: 8) {
                    Label(string("eta"), systemImage: "calendar")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                    Text("·").foregroundStyle(.white.opacity(0.5))
                    Label(string("carrier"), systemImage: "truck.box.fill")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.85))
                }
            }
            .padding(18)
            .background(
                LinearGradient(
                    colors: statusColors.gradient,
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )

            // ── Bottom: items + action ────────────────────────────────
            VStack(alignment: .leading, spacing: 12) {
                if case .array(let items)? = data["items"], !items.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Items")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ForEach(items.indices, id: \.self) { i in
                            if case .string(let name) = items[i] {
                                HStack(spacing: 8) {
                                    Circle()
                                        .fill(statusColors.0)
                                        .frame(width: 6, height: 6)
                                    Text(name)
                                        .font(.subheadline)
                                }
                            }
                        }
                    }
                    Divider()
                }

                Button(action: onRefresh) {
                    Label("Refresh status", systemImage: "arrow.clockwise")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(
                            LinearGradient(
                                colors: statusColors.gradient,
                                startPoint: .leading,
                                endPoint: .trailing
                            ),
                            in: RoundedRectangle(cornerRadius: 12)
                        )
                }
            }
            .padding(18)
            .background(Color(uiColor: .systemBackground))
        }
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .shadow(color: statusColors.0.opacity(0.3), radius: 12, x: 0, y: 6)
    }

    // MARK: - Status palette

    /// Returns (accent color, gradient stops)
    private var statusColors: (Color, gradient: [Color]) {
        switch string("status").lowercased() {
        case "in transit":
            return (.green, [Color(red: 0.1, green: 0.7, blue: 0.4), Color(red: 0.0, green: 0.5, blue: 0.3)])
        case "out for delivery":
            return (.mint, [.mint, Color(red: 0.0, green: 0.6, blue: 0.5)])
        case "delivered":
            return (.blue, [Color(red: 0.2, green: 0.5, blue: 1.0), Color(red: 0.1, green: 0.3, blue: 0.8)])
        case "processing", "pending":
            return (.orange, [Color(red: 1.0, green: 0.6, blue: 0.1), Color(red: 0.9, green: 0.4, blue: 0.0)])
        case "cancelled", "returned":
            return (.red, [Color(red: 0.9, green: 0.2, blue: 0.2), Color(red: 0.7, green: 0.1, blue: 0.1)])
        default:
            return (.indigo, [.indigo, .purple])
        }
    }

    private var statusIcon: String {
        switch string("status").lowercased() {
        case "in transit":         return "shippingbox.fill"
        case "out for delivery":   return "bicycle"
        case "delivered":          return "checkmark.seal.fill"
        case "processing","pending": return "gearshape.fill"
        case "cancelled","returned": return "xmark.circle.fill"
        default:                   return "shippingbox.fill"
        }
    }

    private func string(_ key: String) -> String {
        if case .string(let v)? = data[key] { return v }
        return ""
    }
}
