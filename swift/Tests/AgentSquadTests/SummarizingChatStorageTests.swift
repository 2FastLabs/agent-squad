import Foundation
import Testing

@testable import AgentSquad

@Suite struct SummarizingChatStorageTests {
    private func user(_ t: String) -> ConversationMessage { ConversationMessage(role: .user, text: t) }
    private func assistant(_ t: String) -> ConversationMessage { ConversationMessage(role: .assistant, text: t) }
    private func makeHistory(_ numPairs: Int) -> [ConversationMessage] {
        (0..<numPairs).flatMap { i in [user("User \(i + 1)"), assistant("Assistant \(i + 1)")] }
    }
    private func texts(_ messages: [ConversationMessage]) -> [String] {
        messages.map { $0.parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined() }
    }

    // MARK: - Test helpers

    /// Actor-based counter — safe to capture and mutate in @Sendable closures.
    private actor Counter {
        private(set) var value = 0
        func increment() { value += 1 }
    }

    /// Actor-based slot for capturing a value from inside a @Sendable closure.
    private actor Captured<T: Sendable> {
        private(set) var value: T?
        func set(_ v: T) { value = v }
    }

    // MARK: - SpyStorage

    private actor SpyStorage: ChatStorage {
        var storedMessages: [ConversationMessage] = []
        private(set) var fetchCallCount = 0
        private(set) var saveMessagesBatches: [[ConversationMessage]] = []

        init(messages: [ConversationMessage] = []) {
            storedMessages = messages
        }

        func fetch(userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws -> [ConversationMessage] {
            fetchCallCount += 1
            return storedMessages
        }

        func save(_ message: ConversationMessage, userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws {
            storedMessages.append(message)
        }

        func saveMessages(_ messages: [ConversationMessage], userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws {
            saveMessagesBatches.append(messages)
            storedMessages = messages
        }

        func fetchAllChats(userId: String, sessionId: String) async throws -> [ConversationMessage] {
            storedMessages
        }
    }

    // MARK: - Tests

    @Test func belowTriggerReturnsHistoryUnchanged() async throws {
        let inner = InMemoryChatStorage(makeHistory(3))
        let called = Counter()
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { history, keepLast in
            await called.increment()
            return history
        }, triggerAt: 5, keepLast: 2)

        let result = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        #expect(result.count == 6)
        #expect(await called.value == 0)
    }

    @Test func atTriggerBoundaryDoesNotSummarize() async throws {
        // Exactly triggerAt * 2 messages — condition is strictly >, no summarization.
        let inner = InMemoryChatStorage(makeHistory(5)) // 10 = 5 * 2
        let called = Counter()
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { history, keepLast in
            await called.increment()
            return history
        }, triggerAt: 5, keepLast: 2)

        let result = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        #expect(result.count == 10)
        #expect(await called.value == 0)
    }

    @Test func aboveTriggerCallsSummarizer() async throws {
        let inner = InMemoryChatStorage(makeHistory(6)) // 12 > 10
        let callCount = Counter()
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { history, keepLast in
            await callCount.increment()
            return Array(history.suffix(keepLast * 2))
        }, triggerAt: 5, keepLast: 2)

        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        #expect(await callCount.value == 1)
    }

    @Test func summarizerReceivesFullHistory() async throws {
        let history = makeHistory(6)
        let inner = InMemoryChatStorage(history)
        let receivedCount = Captured<Int>()
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { received, keepLast in
            await receivedCount.set(received.count)
            return Array(received.suffix(keepLast * 2))
        }, triggerAt: 5, keepLast: 2)

        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        #expect(await receivedCount.value == 12)
    }

    @Test func summarizerReceivesKeepLast() async throws {
        let inner = InMemoryChatStorage(makeHistory(6))
        let receivedKeepLast = Captured<Int>()
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { history, keepLast in
            await receivedKeepLast.set(keepLast)
            return Array(history.suffix(keepLast * 2))
        }, triggerAt: 5, keepLast: 3)

        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        #expect(await receivedKeepLast.value == 3)
    }

    @Test func fetchReturnsCompressedResult() async throws {
        let inner = InMemoryChatStorage(makeHistory(6))
        let compressed = [user("Summary of conversation")]
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { _, _ in compressed }, triggerAt: 5, keepLast: 2)

        let result = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        #expect(texts(result) == ["Summary of conversation"])
    }

    @Test func noWriteBackToBaseStoreAfterSummarization() async throws {
        // saveMessages in all bundled stores appends rather than replaces, so
        // writing back the compressed result would corrupt the history.
        // The wrapper uses only the in-memory cache — base store is not written.
        let spy = SpyStorage(messages: makeHistory(6))
        let compressed = [user("Summary")]
        let storage = SummarizingChatStorage(wrapping: spy, summarizer: { _, _ in compressed }, triggerAt: 5, keepLast: 2)

        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        let batches = await spy.saveMessagesBatches
        #expect(batches.count == 0)
    }

    @Test func subsequentFetchUsesCachedResult() async throws {
        let spy = SpyStorage(messages: makeHistory(6))
        let callCount = Counter()
        let storage = SummarizingChatStorage(wrapping: spy, summarizer: { _, _ in
            await callCount.increment()
            return [ConversationMessage(role: .user, text: "Summary")]
        }, triggerAt: 5, keepLast: 2)

        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        // Summarizer called only once — second fetch hits the internal cache.
        #expect(await callCount.value == 1)
        let fetchCount = await spy.fetchCallCount
        #expect(fetchCount == 1)
    }

    @Test func fetchAllChatsNeverIntercepted() async throws {
        let inner = InMemoryChatStorage(makeHistory(6))
        let called = Counter()
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { history, keepLast in
            await called.increment()
            return history
        }, triggerAt: 5, keepLast: 2)

        let result = try await storage.fetchAllChats(userId: "u", sessionId: "s")

        #expect(await called.value == 0)
        #expect(result.count == 12)
    }

    @Test func saveDelegatesToBase() async throws {
        let spy = SpyStorage()
        let storage = SummarizingChatStorage(wrapping: spy, summarizer: { h, _ in h }, triggerAt: 5, keepLast: 2)

        try await storage.save(user("Hello"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        let stored = await spy.storedMessages
        #expect(texts(stored) == ["Hello"])
    }

    @Test func saveMessagesDelegatesToBase() async throws {
        let spy = SpyStorage()
        let storage = SummarizingChatStorage(wrapping: spy, summarizer: { h, _ in h }, triggerAt: 5, keepLast: 2)

        try await storage.saveMessages([user("A"), assistant("B")], userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        let batches = await spy.saveMessagesBatches
        #expect(batches.count == 1)
        #expect(texts(batches[0]) == ["A", "B"])
    }

    @Test func saveInvalidatesCacheSoNextFetchHitsBaseStore() async throws {
        let spy = SpyStorage(messages: makeHistory(6))
        let callCount = Counter()
        let storage = SummarizingChatStorage(wrapping: spy, summarizer: { _, _ in
            await callCount.increment()
            return [ConversationMessage(role: .user, text: "Summary")]
        }, triggerAt: 5, keepLast: 2)

        // First fetch: triggers summarization, caches result. spy.fetch called once.
        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(await spy.fetchCallCount == 1)

        // Second fetch: served from cache — spy.fetch NOT called again.
        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(await spy.fetchCallCount == 1)

        // A save invalidates the cache.
        try await storage.save(user("New message"), userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)

        // Next fetch must re-hit the base store (cache was cleared).
        _ = try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        #expect(await spy.fetchCallCount == 2)
    }

    @Test func summarizerThrowingPropagates() async throws {
        struct SummarizationFailed: Error {}
        let inner = InMemoryChatStorage(makeHistory(6))
        let storage = SummarizingChatStorage(wrapping: inner, summarizer: { _, _ in
            throw SummarizationFailed()
        }, triggerAt: 5, keepLast: 2)

        await #expect(throws: SummarizationFailed.self) {
            try await storage.fetch(userId: "u", sessionId: "s", agentId: "a", maxMessages: nil)
        }
    }
}
