import { ChatStorage } from "./chatStorage";
import { ConversationMessage } from "../types";

/**
 * Async callable that compresses a conversation buffer.
 *
 * Receives the current buffer and the number of recent pairs to keep verbatim.
 * Must return the compressed history (typically a summary message followed by
 * the last `keepLast` pairs).
 */
export type ChatSummarizer = (
  history: ConversationMessage[],
  keepLast: number
) => Promise<ConversationMessage[]>;

/**
 * A `ChatStorage` wrapper that keeps agent context small via summarization.
 *
 * Raw messages are always written to the inner storage untouched — they remain
 * available for analytics, audit, or replay via `fetchAllChats`. The summarizer
 * only affects what the agent sees through `fetchChat`.
 *
 * **How it works**
 *
 * An in-memory buffer is maintained per (userId, sessionId, agentId) slot:
 *
 * - The buffer is **activated lazily** on the first `fetchChat` call that finds
 *   history above the threshold. Before that, all operations are pure
 *   delegations to the inner storage.
 *
 * - Once the buffer is active, every `saveChatMessage` appends the new message
 *   to it and, if the buffer exceeds the threshold again, calls the summarizer
 *   **immediately** — so the next `fetchChat` is always fast.
 *
 * - `fetchAllChats` is never intercepted: raw full history is always available.
 *
 * @example
 * ```typescript
 * const storage = new SummarizingChatStorage(
 *   new InMemoryChatStorage(),
 *   async (history, keepLast) => {
 *     const old = history.slice(0, -keepLast * 2);
 *     const recent = history.slice(-keepLast * 2);
 *     const summary = await callLlmToSummarize(old);
 *     return [{ role: 'user', content: [{ text: `[Summary]: ${summary}` }] }, ...recent];
 *   },
 *   20,  // triggerAt
 *   2,   // keepLast
 * );
 * ```
 */
export class SummarizingChatStorage extends ChatStorage {
  /**
   * Per-(userId, sessionId, agentId) in-memory buffer of the current compressed
   * history. A missing key means the buffer is not yet active — saves are pure
   * delegations until the first fetch crosses the threshold.
   */
  private readonly buffers: Map<string, ConversationMessage[]> = new Map();

  constructor(
    private readonly storage: ChatStorage,
    private readonly summarizer: ChatSummarizer,
    private readonly triggerAt: number = 20,
    private readonly keepLast: number = 2
  ) {
    super();
  }

  private key(userId: string, sessionId: string, agentId: string): string {
    return `${userId}#${sessionId}#${agentId}`;
  }

  private async compressIfNeeded(key: string): Promise<void> {
    const buf = this.buffers.get(key);
    if (buf && buf.length > this.triggerAt * 2) {
      this.buffers.set(key, await this.summarizer(buf, this.keepLast));
    }
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    const key = this.key(userId, sessionId, agentId);
    const buf = this.buffers.get(key);
    if (buf !== undefined) {
      buf.push(newMessage);
      await this.compressIfNeeded(key);
    }
    return this.storage.saveChatMessage(userId, sessionId, agentId, newMessage, maxHistorySize);
  }

  async fetchChat(
    userId: string,
    sessionId: string,
    agentId: string,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    const key = this.key(userId, sessionId, agentId);

    // Buffer is active — return it directly (no storage read, no LLM call).
    const buf = this.buffers.get(key);
    if (buf !== undefined) {
      return buf;
    }

    // Cold start: load raw history from the inner store.
    const history = await this.storage.fetchChat(userId, sessionId, agentId, maxHistorySize);

    if (history.length > this.triggerAt * 2) {
      const compressed = await this.summarizer(history, this.keepLast);
      this.buffers.set(key, compressed);
      return compressed;
    }

    return history;
  }

  async fetchAllChats(
    userId: string,
    sessionId: string
  ): Promise<ConversationMessage[]> {
    // Never intercepted — raw history always available for analytics/audit.
    return this.storage.fetchAllChats(userId, sessionId);
  }
}
