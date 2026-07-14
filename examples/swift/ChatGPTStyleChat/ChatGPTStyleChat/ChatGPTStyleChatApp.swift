import SwiftUI

@main
struct ChatGPTStyleChatApp: App {
    @StateObject private var vm = ChatViewModel()

    var body: some Scene {
        WindowGroup {
            ChatView(vm: vm)
        }
    }
}
