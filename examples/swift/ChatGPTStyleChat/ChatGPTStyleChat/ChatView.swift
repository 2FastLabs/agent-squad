import SwiftUI

struct ChatView: View {
    @ObservedObject var vm: ChatViewModel
    @State private var draft = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messages
            Divider()
            composer
        }
    }

    private var header: some View {
        HStack {
            Text("Shop").font(.headline)
            Spacer()
            Text(vm.showWidgets ? "Text + widgets" : "Text only")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Widgets", isOn: $vm.showWidgets).labelsHidden()
        }
        .padding()
    }

    private var messages: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(vm.items) { item in
                    VStack(alignment: item.isUser ? .trailing : .leading, spacing: 8) {
                        if let payload = item.widget {
                            WidgetView(payload: payload) { tool, args in
                                vm.handleWidgetTool(tool, arguments: args, for: item.id)
                            }
                        }
                        if !item.text.isEmpty {
                            Text(item.text)
                                .padding(10)
                                .background(item.isUser ? Color.blue.opacity(0.15)
                                                        : Color.gray.opacity(0.12))
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: item.isUser ? .trailing : .leading)
                }
            }
            .padding()
        }
    }

    private var composer: some View {
        HStack {
            TextField("Try: where is my order #1234?", text: $draft)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
            Button("Send", action: send)
        }
        .padding()
    }

    private func send() {
        vm.send(draft)
        draft = ""
    }
}
