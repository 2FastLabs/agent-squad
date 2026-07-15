import SwiftUI
import AgentSquad

struct ChatView: View {
    @ObservedObject var vm: ChatViewModel
    @State private var draft = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            messages
            composer
        }
        .background(Color(uiColor: .systemGroupedBackground))
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Shop Assistant")
                    .font(.title3).fontWeight(.semibold)
                Text("Powered by Agent Squad")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 6) {
                Image(systemName: vm.showWidgets ? "rectangle.on.rectangle" : "text.bubble")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Toggle("", isOn: $vm.showWidgets)
                    .labelsHidden()
                    .tint(Color(red: 0.0, green: 0.48, blue: 1.0))
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(.regularMaterial)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    // MARK: Messages

    private var messages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(vm.items) { item in
                        MessageRow(item: item) { tool, args in
                            vm.handleWidgetTool(tool, arguments: args, for: item.id)
                        }
                        .id(item.id)
                    }
                }
                .padding(.vertical, 12)
            }
            .onChange(of: vm.items.count) { _, _ in
                if let last = vm.items.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    // MARK: Composer

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message", text: $draft)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 22)
                        .fill(Color(uiColor: .secondarySystemBackground))
                        .overlay(RoundedRectangle(cornerRadius: 22)
                            .strokeBorder(Color(uiColor: .separator), lineWidth: 0.5))
                )
                .focused($inputFocused)
                .onSubmit(send)
                .submitLabel(.send)

            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                     ? Color(uiColor: .tertiaryLabel) : Color(red: 0.0, green: 0.48, blue: 1.0))
            }
            .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .animation(.easeInOut(duration: 0.15), value: draft.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.regularMaterial)
        .overlay(alignment: .top) { Divider() }
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        vm.send(text)
        draft = ""
    }
}

// MARK: - Message row

private struct MessageRow: View {
    let item: ChatItem
    let onTool: (String, JSONValue) -> Void

    var body: some View {
        VStack(alignment: item.isUser ? .trailing : .leading, spacing: 6) {
            if let payload = item.widget {
                WidgetView(payload: payload, onAppTool: onTool)
                    .padding(.horizontal, 16)
            }

            if !item.text.isEmpty {
                Text(item.text)
                    .font(.body)
                    .foregroundStyle(item.isUser ? .white : .primary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        item.isUser
                            ? AnyShapeStyle(Color(red: 0.0, green: 0.48, blue: 1.0))
                            : AnyShapeStyle(Color(uiColor: .secondarySystemBackground))
                    )
                    .clipShape(BubbleShape(isUser: item.isUser))
                    .shadow(color: .black.opacity(0.06), radius: 4, x: 0, y: 2)
                    .padding(item.isUser ? .leading : .trailing, 60)
                    .padding(.horizontal, 16)
            }
        }
        .frame(maxWidth: .infinity, alignment: item.isUser ? .trailing : .leading)
        .padding(.vertical, 2)
    }
}

// MARK: - Bubble shape (iMessage style tail)

private struct BubbleShape: Shape {
    let isUser: Bool

    func path(in rect: CGRect) -> Path {
        let r: CGFloat = 18
        var path = Path()
        path.addRoundedRect(in: rect, cornerSize: CGSize(width: r, height: r))
        return path
    }
}
