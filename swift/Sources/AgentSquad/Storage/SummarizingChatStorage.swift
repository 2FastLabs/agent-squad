import Foundation

/// Async callable that compresses a conversation buffer.
///
/// Receives the current buffer and the number of recent pairs to keep verbatim.
/// Must return the compressed history — typically a summary message followed by
/// the last `keepLast` pairs.
public typealias ChatSummarizer = @Sendable (
    _ history: [ConversationMessage],
    _ keepLast: Int
) async throws -> [ConversationMessage]

// MARK: - Internal buffer actor

private actor BufferStore {
    private var buffers: [String: [ConversationMessage]] = [:]

    func get(key: String) -> [ConversationMessage]? {
        buffers[key]
    }

    func set(key: String, messages: [ConversationMessage]) {
        buffers[key] = messages
    }

    func append(key: String, message: ConversationMessage) {
        buffers[key]?.append(message)
    }
}

// MARK: - SummarizingChatStorage

/// A `ChatStorage` wrapper that keeps agent context small via summarization.
///
/// Raw messages are always written to the base store untouched — they remain
/// available for analytics, audit, or replay via `fetchAllChats`. The
/// summarizer only affects what the agent sees through `fetch`.
///
/// **How it works**
///
/// An in-memory buffer is maintained per (userId, sessionId, agentId) slot:
///
/// - The buffer is **activated lazily** on the first `fetch` call that finds
///   history above the threshold. Before that, all operations are pure
///   delegations to the base store.
///
/// - Once the buffer is active, every `save` appends the new message to it
///   and, if the buffer exceeds the threshold again, calls the summarizer
///   **immediately** — so the next `fetch` is always fast.
///
/// - `fetchAllChats` is never intercepted: raw full history is always available.
///
/// ```swift
/// let storage = SummarizingChatStorage(
///     wrapping: InMemoryChatStorage(),
///     summarizer: { history, keepLast in
///         let old = Array(history.dropLast(keepLast * 2))
///         let recent = Array(history.suffix(keepLast * 2))
///         let summaryText = try await myLLM.summarize(old)
///         let summary = ConversationMessage(role: .user, text: "[Summary]: \(summaryText)")
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
    private let bufferStore: BufferStore

    /// - Parameters:
    ///   - base: The inner `ChatStorage` to wrap.
    ///   - summarizer: Async callable that compresses the buffer.
    ///   - triggerAt: Number of message **pairs** above which the buffer is
    ///     compressed. Defaults to 20.
    ///   - keepLast: Number of most-recent message pairs to keep verbatim.
    ///     Defaults to 2.
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
        self.bufferStore = BufferStore()
    }

    public func fetch(
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws -> [ConversationMessage] {
        let key = bufferKey(userId: userId, sessionId: sessionId, agentId: agentId)

        // Buffer is active — return it directly (no base read, no LLM call).
        if let buf = await bufferStore.get(key: key) {
            return buf
        }

        // Cold start: load raw history from the base store.
        let history = try await base.fetch(
            userId: userId, sessionId: sessionId, agentId: agentId, maxMessages: maxMessages
        )

        guard history.count > triggerAt * 2 else {
            return history
        }

        let compressed = try await summarizer(history, keepLast)
        await bufferStore.set(key: key, messages: compressed)
        return compressed
    }

    public func save(
        _ message: ConversationMessage,
        userId: String,
        sessionId: String,
        agentId: String,
        maxMessages: Int?
    ) async throws {
        let key = bufferKey(userId: userId, sessionId: sessionId, agentId: agentId)
        if await bufferStore.get(key: key) != nil {
            await bufferStore.append(key: key, message: message)
            try await compressIfNeeded(key: key)
        }
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
        let key = bufferKey(userId: userId, sessionId: sessionId, agentId: agentId)
        if await bufferStore.get(key: key) != nil {
            for message in messages {
                await bufferStore.append(key: key, message: message)
            }
            try await compressIfNeeded(key: key)
        }
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
        // Never intercepted — raw history always available for analytics/audit.
        try await base.fetchAllChats(userId: userId, sessionId: sessionId)
    }

    // MARK: - Private

    private func bufferKey(userId: String, sessionId: String, agentId: String) -> String {
        "\(userId)#\(sessionId)#\(agentId)"
    }

    private func compressIfNeeded(key: String) async throws {
        guard let buf = await bufferStore.get(key: key), buf.count > triggerAt * 2 else { return }
        let compressed = try await summarizer(buf, keepLast)
        await bufferStore.set(key: key, messages: compressed)
    }
}
