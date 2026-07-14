import Foundation

/// Compresses a conversation history into a shorter form.
///
/// Receives the full history and the number of most-recent message pairs to
/// keep verbatim. Must return the compressed history — typically a summary
/// message followed by the last `keepLast` pairs.
public typealias ChatSummarizer = @Sendable (
    _ history: [ConversationMessage],
    _ keepLast: Int
) async throws -> [ConversationMessage]

// MARK: - Internal cache actor

private actor SummarizationCache {
    private var store: [String: [ConversationMessage]] = [:]

    func get(key: String) -> [ConversationMessage]? {
        store[key]
    }

    func set(key: String, messages: [ConversationMessage]) {
        store[key] = messages
    }

    func invalidate(key: String) {
        store.removeValue(forKey: key)
    }
}

// MARK: - SummarizingChatStorage

/// A `ChatStorage` wrapper that automatically compresses long conversation histories.
///
/// Wraps any `ChatStorage` implementation. On every `fetch` call, if the
/// returned history exceeds `triggerAt` message pairs the user-supplied
/// `summarizer` callable is invoked. The compressed result is cached in memory
/// so subsequent fetches are fast without hitting the base store again.
///
/// `fetchAllChats` is never intercepted — the classifier always sees the full
/// cross-agent history unmodified.
///
/// A `save` or `saveMessages` call invalidates the internal cache for that
/// conversation slot so the next `fetch` re-evaluates from the base store.
///
/// ```swift
/// let storage = SummarizingChatStorage(
///     wrapping: FileChatStorage(),
///     summarizer: { history, keepLast in
///         let old = Array(history.dropLast(keepLast * 2))
///         let recent = Array(history.suffix(keepLast * 2))
///         let summaryText = try await myLLM.summarize(old)
///         let summary = ConversationMessage(role: .user, content: "[Summary]: \(summaryText)")
///         return [summary] + recent
///     },
///     triggerAt: 20,
///     keepLast: 2
/// )
/// ```
public struct SummarizingChatStorage: ChatStorage {
    private let base: any ChatStorage
    private let summarizer: ChatSummarizer
    private let triggerAt: Int
    private let keepLast: Int
    private let cache: SummarizationCache

    /// - Parameters:
    ///   - base: The inner `ChatStorage` to wrap.
    ///   - summarizer: Async callable that compresses the history.
    ///   - triggerAt: Number of message **pairs** above which summarization is
    ///     triggered. Defaults to 20.
    ///   - keepLast: Number of most-recent message pairs to keep verbatim,
    ///     passed to the summarizer. Defaults to 2.
    public init(
        wrapping base: any ChatStorage,
        summarizer: @escaping ChatSummarizer,
        triggerAt: Int = 20,
        keepLast: Int = 2
    ) {
        self.base = base
        self.summarizer = summarizer
        self.triggerAt = triggerAt
        self.keepLast = keepLast
        self.cache = SummarizationCache()
    }

    public func fetch(
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws -> [ConversationMessage] {
        let key = cacheKey(userId: userId, sessionId: sessionId, agentId: agentId)

        // Return cached compressed history if available.
        if let cached = await cache.get(key: key) {
            return cached
        }

        let history = try await base.fetch(
            userId: userId, sessionId: sessionId, agentId: agentId, maxMessages: maxMessages
        )

        guard history.count > triggerAt * 2 else {
            return history
        }

        let compressed = try await summarizer(history, keepLast)

        // Cache the compressed result in memory. All bundled stores use
        // append semantics in saveMessages, so writing back would corrupt
        // the history by adding the summary on top of the original messages.
        await cache.set(key: key, messages: compressed)

        return compressed
    }

    public func save(
        _ message: ConversationMessage,
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws {
        await cache.invalidate(key: cacheKey(userId: userId, sessionId: sessionId, agentId: agentId))
        try await base.save(
            message,
            userId: userId, sessionId: sessionId, agentId: agentId,
            maxMessages: maxMessages
        )
    }

    public func saveMessages(
        _ messages: [ConversationMessage],
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws {
        await cache.invalidate(key: cacheKey(userId: userId, sessionId: sessionId, agentId: agentId))
        try await base.saveMessages(
            messages,
            userId: userId, sessionId: sessionId, agentId: agentId,
            maxMessages: maxMessages
        )
    }

    public func fetchAllChats(
        userId: String,
        sessionId: String
    ) async throws -> [ConversationMessage] {
        // Never intercepted — classifier must see the full cross-agent history.
        try await base.fetchAllChats(userId: userId, sessionId: sessionId)
    }

    // MARK: - Private

    private func cacheKey(userId: String, sessionId: String, agentId: String) -> String {
        "\(userId)#\(sessionId)#\(agentId)"
    }
}
