import { ChatStorage } from "./chatStorage";
import { ConversationMessage } from "../types";

/**
 * A callable that compresses a conversation history.
 *
 * Receives the full history and the number of recent pairs to keep verbatim.
 * Must return the compressed history (summary message(s) + recent pairs).
 */
export type ChatSummarizer = (
  history: ConversationMessage[],
  keepLast: number
) => Promise<ConversationMessage[]>;

/**
 * A `ChatStorage` wrapper that automatically compresses long conversation histories.
 *
 * Wraps any `ChatStorage` implementation. On every `fetchChat` call, if the
 * returned history exceeds `triggerAt` message pairs the user-supplied
 * `summarizer` callable is invoked. The compressed result is cached internally
 * so subsequent fetches are fast without re-calling the summarizer.
 *
 * `fetchAllChats` is never intercepted — the classifier always sees the full
 * cross-agent history unmodified.
 *
 * @example
 * ```typescript
 * import { SummarizingChatStorage, InMemoryChatStorage } from 'agent-squad';
 *
 * async function mySummarizer(history, keepLast) {
 *   const old = history.slice(0, -keepLast * 2);
 *   const recent = history.slice(-keepLast * 2);
 *   const summary = await callLlmToSummarize(old);
 *   return [
 *     { role: 'user', content: [{ text: `[Summary]: ${summary}` }] },
 *     ...recent,
 *   ];
 * }
 *
 * const storage = new SummarizingChatStorage(
 *   new InMemoryChatStorage(),
 *   mySummarizer,
 *   20,  // triggerAt: summarize when history exceeds 20 pairs
 *   2,   // keepLast: keep the 2 most recent pairs verbatim
 * );
 * ```
 */
export class SummarizingChatStorage extends ChatStorage {
  /**
   * Internal cache: key → compressed history.
   * After summarization the compressed result is stored here so subsequent
   * `fetchChat` calls return immediately without hitting the inner storage or
   * re-invoking the summarizer. A `saveChatMessage` call invalidates the entry
   * so the next fetch re-evaluates from the inner store.
   */
  private readonly cache: Map<string, ConversationMessage[]> = new Map();

  /**
   * @param storage  The inner `ChatStorage` to wrap.
   * @param summarizer  Async callable that compresses history.
   * @param triggerAt  Number of message **pairs** above which summarization
   *   is triggered. Defaults to 20.
   * @param keepLast  Number of most-recent message pairs to keep verbatim.
   *   Passed to the summarizer as the second argument. Defaults to 2.
   */
  constructor(
    private readonly storage: ChatStorage,
    private readonly summarizer: ChatSummarizer,
    private readonly triggerAt: number = 20,
    private readonly keepLast: number = 2
  ) {
    super();
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    // Invalidate cache so the next fetchChat re-evaluates from the inner store.
    this.cache.delete(this.cacheKey(userId, sessionId, agentId, maxHistorySize));
    return this.storage.saveChatMessage(userId, sessionId, agentId, newMessage, maxHistorySize);
  }

  async fetchChat(
    userId: string,
    sessionId: string,
    agentId: string,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    const key = this.cacheKey(userId, sessionId, agentId, maxHistorySize);

    // Return cached compressed history if available.
    const cached = this.cache.get(key);
    if (cached !== undefined) {
      return cached;
    }

    const history = await this.storage.fetchChat(userId, sessionId, agentId, maxHistorySize);

    if (history.length > this.triggerAt * 2) {
      const compressed = await this.summarizer(history, this.keepLast);
      this.cache.set(key, compressed);
      return compressed;
    }

    return history;
  }

  async fetchAllChats(
    userId: string,
    sessionId: string
  ): Promise<ConversationMessage[]> {
    // Never intercepted — classifier must see the full cross-agent history.
    return this.storage.fetchAllChats(userId, sessionId);
  }

  private cacheKey(userId: string, sessionId: string, agentId: string, maxHistorySize?: number): string {
    return `${userId}#${sessionId}#${agentId}#${maxHistorySize ?? 'nil'}`;
  }
}
