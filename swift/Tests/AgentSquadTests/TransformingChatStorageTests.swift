import Foundation
import Testing

@testable import AgentSquad

@Suite struct TransformingChatStorageTests {
    private func user(_ t: String) -> ConversationMessage { ConversationMessage(role: .user, text: t) }
    private func assistant(_ t: String) -> ConversationMessage { ConversationMessage(role: .assistant, text: t) }
    private func texts(_ messages: [ConversationMessage]) -> [String] {
        messages.map { $0.parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined() }
    }

    @Test func transformsMessagesBeforeSaving() async throws {
        let store = TransformingChatStorage(wrapping: InMemoryChatStorage()) { message in
            message.mappingText { $0.replacingOccurrences(of: "4242 4242 4242 4242", with: "[CARD]") }
        }
        try await store.save(user("my card is 4242 4242 4242 4242"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        try await store.saveMessages([assistant("noted 4242 4242 4242 4242")], userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        let history = try await store.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(texts(history) == ["my card is [CARD]", "noted [CARD]"])   // scrubbed form persisted, reads pass through
    }

    @Test func nilDropsTheMessage() async throws {
        let store = TransformingChatStorage(wrapping: InMemoryChatStorage()) { message in
            message.text.contains("secret") ? nil : message
        }
        try await store.save(user("hello"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        try await store.save(assistant("the secret is 42"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        let history = try await store.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(texts(history) == ["hello"])
    }

    @Test func droppingAMessageCanSkipItsCounterpartToo() async throws {
        // The documented pairing side-effect: dropping the assistant reply makes the next user
        // message consecutive-same-role in the base store, which then skips it.
        let store = TransformingChatStorage(wrapping: InMemoryChatStorage()) { message in
            message.text.contains("secret") ? nil : message
        }
        try await store.save(user("hello"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        try await store.saveMessages([assistant("the secret is 42"), user("bye")], userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        let history = try await store.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(texts(history) == ["hello"])   // "bye" skipped — prefer redacting over dropping
    }

    /// Records `saveMessages` calls — proves the wrapper never bothers the base with an empty batch
    /// (a spurious write in `FileChatStorage`, unknown side effects in custom stores).
    private actor SpyStorage: ChatStorage {
        private(set) var saveBatches: [[ConversationMessage]] = []
        func fetch(userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws -> [ConversationMessage] { [] }
        func save(_ message: ConversationMessage, userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws {}
        func saveMessages(_ messages: [ConversationMessage], userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws {
            saveBatches.append(messages)
        }
        func fetchAllChats(userId: String, sessionId: String) async throws -> [ConversationMessage] { [] }
    }

    @Test func allDroppedSkipsTheBaseSaveEntirely() async throws {
        let spy = SpyStorage()
        let store = TransformingChatStorage(wrapping: spy) { _ in nil }
        try await store.saveMessages([user("a"), assistant("b")], userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(await spy.saveBatches.isEmpty)   // nothing survived → the base is never called
    }

    @Test func aThrowingTransformFailsTheSave() async throws {
        struct ScrubFailed: Error {}
        let spy = SpyStorage()
        let store = TransformingChatStorage(wrapping: spy) { _ in throw ScrubFailed() }
        await #expect(throws: ScrubFailed.self) {
            try await store.save(user("pii"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        }
        #expect(await spy.saveBatches.isEmpty)   // fails loudly, never persists unscrubbed
    }

    @Test func mappingTextAlsoCoversAudioTranscripts() async throws {
        let message = ConversationMessage(role: .user, parts: [.audioTranscript("my card is 4242")])
        let mapped = message.mappingText { $0.replacingOccurrences(of: "4242", with: "[CARD]") }
        #expect(mapped.parts == [.audioTranscript("my card is [CARD]")])
    }

    @Test func transformKeepsMessageIdentityAndRole() async throws {
        let original = user("call me on 0612345678")
        let scrubbed = original.mappingText { $0.replacingOccurrences(of: "0612345678", with: "[PHONE]") }
        #expect(scrubbed.id == original.id)
        #expect(scrubbed.role == original.role)
        #expect(scrubbed.timestamp == original.timestamp)
        #expect(scrubbed.text == "call me on [PHONE]")
    }

    @Test func mappingTextLeavesNonTextPartsAlone() async throws {
        let widget = ContentPart.widget(UIPayload(resourceURI: "ui://odds", mimeType: "text/html"))
        let message = ConversationMessage(role: .assistant, parts: [.text("secret"), widget])
        let mapped = message.mappingText { _ in "[REDACTED]" }
        #expect(mapped.parts.count == 2)
        #expect(mapped.text == "[REDACTED]")
        #expect(mapped.parts[1] == widget)
    }

    @Test func docExampleScrubPIICompilesAndWorks() async throws {
        // The `scrubPII` worked example from docs/swift/storage/built-in/transforming.md, verbatim.
        let scrubPII: MessageTransform = { message in
            message.mappingText { text in
                var scrubbed = text
                scrubbed = scrubbed.replacing(#/\b[A-Z]{2}\d{2}(?: ?\d{4}){4,7}\b/#, with: "[IBAN]")
                scrubbed = scrubbed.replacing(#/\b(?:\d[ -]?){13,19}\b/#, with: "[CARD]")
                scrubbed = scrubbed.replacing(#/[\w.+-]+@[\w-]+\.[\w.]+/#, with: "[EMAIL]")
                return scrubbed
            }
        }
        let store = TransformingChatStorage(wrapping: InMemoryChatStorage(), transform: scrubPII)
        try await store.save(
            user("pay XX12 3456 7890 1234 5678 or 4242 4242 4242 4242, mail c.croitoru@example.com"),
            userId: "u", sessionId: "s", agentId: "a", maxMessages: nil
        )
        let history = try await store.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(texts(history) == ["pay [IBAN] or [CARD], mail [EMAIL]"])
    }

    @Test func readsPassThroughUnchanged() async throws {
        let store = TransformingChatStorage(wrapping: InMemoryChatStorage([user("q"), assistant("a")])) { _ in nil }
        // A drop-everything transform must not affect what fetch/fetchAllChats return.
        #expect(texts(try await store.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)) == ["q", "a"])
        #expect(texts(try await store.fetchAllChats(userId: "u", sessionId: "s")) == ["q", "a"])
    }
}
