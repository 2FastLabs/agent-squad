import Foundation

/// A message transform run before persistence: strip PII, redact tokens, clip oversized payloads —
/// or return `nil` to drop the message entirely (not persisted). Throwing fails the save loudly —
/// the alternative to silently persisting unscrubbed data when a scrubber errors.
public typealias MessageTransform = @Sendable (ConversationMessage) async throws -> ConversationMessage?

/// Wraps any `ChatStorage` and runs a `MessageTransform` on every message before it reaches the
/// wrapped store — the seam for PII scrubbing and message shaping at the persistence boundary.
/// Reads pass through untouched: history returns what was persisted, i.e. the transformed form
/// (which is the point — scrubbed data never re-enters prompts from storage).
///
/// ```swift
/// let store = TransformingChatStorage(wrapping: FileChatStorage()) { message in
///     message.mappingText { $0.replacing(iban, with: "[IBAN]") }
/// }
/// ```
///
/// Dropping a message (`nil`) can leave an unpaired exchange: stores skip a save whose role
/// repeats the last stored message's, so the counterpart of a dropped message may be skipped too.
public struct TransformingChatStorage: ChatStorage {
    private let base: any ChatStorage
    private let transform: MessageTransform

    public init(wrapping base: any ChatStorage, transform: @escaping MessageTransform) {
        self.base = base
        self.transform = transform
    }

    public func fetch(
        userId: String, sessionId: String, agentId: String, maxMessages: Int?
    ) async throws -> [ConversationMessage] {
        try await base.fetch(userId: userId, sessionId: sessionId, agentId: agentId, maxMessages: maxMessages)
    }

    public func save(
        _ message: ConversationMessage,
        userId: String, sessionId: String, agentId: String, maxMessages: Int?
    ) async throws {
        guard let transformed = try await transform(message) else { return }
        try await base.save(transformed, userId: userId, sessionId: sessionId, agentId: agentId, maxMessages: maxMessages)
    }

    public func saveMessages(
        _ messages: [ConversationMessage],
        userId: String, sessionId: String, agentId: String, maxMessages: Int?
    ) async throws {
        var transformed: [ConversationMessage] = []
        for message in messages {
            if let message = try await transform(message) { transformed.append(message) }
        }
        guard !transformed.isEmpty else { return }
        try await base.saveMessages(transformed, userId: userId, sessionId: sessionId, agentId: agentId, maxMessages: maxMessages)
    }

    public func fetchAllChats(userId: String, sessionId: String) async throws -> [ConversationMessage] {
        try await base.fetchAllChats(userId: userId, sessionId: sessionId)
    }
}

public extension ConversationMessage {
    /// The same message with every string part — `.text` AND `.audioTranscript` — run through
    /// `transform`: the common PII-scrub shape. Structured parts (`toolCall`/`toolResult` payloads,
    /// widgets) pass through; use a full `MessageTransform` to touch those.
    func mappingText(_ transform: (String) throws -> String) rethrows -> ConversationMessage {
        let parts = try parts.map { part in
            switch part {
            case .text(let text): ContentPart.text(try transform(text))
            case .audioTranscript(let text): ContentPart.audioTranscript(try transform(text))
            default: part
            }
        }
        return ConversationMessage(id: id, role: role, parts: parts, timestamp: timestamp)
    }
}
