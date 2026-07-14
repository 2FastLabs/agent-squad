import { InMemoryChatStorage } from "../../src/storage/memoryChatStorage";
import { SummarizingChatStorage } from "../../src/storage/summarizingChatStorage";
import { ConversationMessage, ParticipantRole } from "../../src/types";

const createMessage = (role: ParticipantRole, text: string): ConversationMessage => ({
  role,
  content: [{ text }],
});

const makeHistory = (numPairs: number): ConversationMessage[] => {
  const messages: ConversationMessage[] = [];
  for (let i = 0; i < numPairs; i++) {
    messages.push(createMessage(ParticipantRole.USER, `User message ${i + 1}`));
    messages.push(createMessage(ParticipantRole.ASSISTANT, `Assistant message ${i + 1}`));
  }
  return messages;
};

const identitySummarizer = async (
  history: ConversationMessage[],
  keepLast: number
): Promise<ConversationMessage[]> => history.slice(-(keepLast * 2));

describe("SummarizingChatStorage", () => {
  let inner: InMemoryChatStorage;

  beforeEach(() => {
    inner = new InMemoryChatStorage();
  });

  test("returns history unchanged when below trigger", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const history = makeHistory(3);
    await inner.saveChatMessage("u", "s", "a", history[0]);
    for (const msg of history.slice(1)) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    const result = await storage.fetchChat("u", "s", "a");

    expect(result).toHaveLength(6);
    expect(summarizer).not.toHaveBeenCalled();
  });

  test("does not call summarizer at exactly the trigger boundary", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    // Seed inner storage directly via save_chat_messages equivalent
    const history = makeHistory(5); // exactly 10 = triggerAt * 2
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    const result = await storage.fetchChat("u", "s", "a");

    expect(result).toHaveLength(10);
    expect(summarizer).not.toHaveBeenCalled();
  });

  test("calls summarizer when history exceeds trigger", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const history = makeHistory(6); // 12 > 10
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    await storage.fetchChat("u", "s", "a");

    expect(summarizer).toHaveBeenCalledTimes(1);
  });

  test("summarizer receives full history as first argument", async () => {
    let receivedHistory: ConversationMessage[] = [];
    const capturingSummarizer = jest.fn(async (history: ConversationMessage[], _keepLast: number) => {
      receivedHistory = history;
      return history.slice(-4);
    });

    const storage = new SummarizingChatStorage(inner, capturingSummarizer, 5, 2);
    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    await storage.fetchChat("u", "s", "a");

    expect(receivedHistory).toHaveLength(12);
  });

  test("summarizer receives keepLast as second argument", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 3);

    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    await storage.fetchChat("u", "s", "a");

    expect(summarizer).toHaveBeenCalledWith(expect.any(Array), 3);
  });

  test("fetchChat returns compressed result after summarization", async () => {
    const compressed = [createMessage(ParticipantRole.USER, "Summary of conversation")];
    const summarizer = jest.fn(async () => compressed);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    const result = await storage.fetchChat("u", "s", "a");

    expect(result).toHaveLength(1);
    expect(result[0].content?.[0]?.text).toBe("Summary of conversation");
  });

  test("returns compressed result on subsequent fetchChat without calling summarizer again", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    await storage.fetchChat("u", "s", "a");
    const result = await storage.fetchChat("u", "s", "a");

    expect(summarizer).toHaveBeenCalledTimes(1);
    expect(result).toHaveLength(4); // keepLast=2 pairs
  });

  test("saveChatMessage invalidates cache so next fetch re-evaluates", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    // First fetch — triggers summarization, caches result
    await storage.fetchChat("u", "s", "a");
    expect(summarizer).toHaveBeenCalledTimes(1);

    // New message invalidates cache
    await storage.saveChatMessage("u", "s", "a", createMessage(ParticipantRole.USER, "New message"));

    // Next fetch re-evaluates from inner store (which now has compressed + new message)
    await storage.fetchChat("u", "s", "a");
    expect(summarizer).toHaveBeenCalledTimes(2);
  });

  test("fetchAllChats delegates to inner storage without interception", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    const result = await storage.fetchAllChats("u", "s");

    expect(summarizer).not.toHaveBeenCalled();
    expect(result).toHaveLength(12);
  });

  test("saveChatMessage delegates to inner storage", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    const msg = createMessage(ParticipantRole.USER, "Hello");
    await storage.saveChatMessage("u", "s", "a", msg);

    const saved = await inner.fetchChat("u", "s", "a");
    expect(saved).toHaveLength(1);
    expect(saved[0].content?.[0]?.text).toBe("Hello");
  });

  test("summarizer error propagates from fetchChat", async () => {
    const failingSummarizer = jest.fn(async () => {
      throw new Error("summarizer failed");
    });
    const storage = new SummarizingChatStorage(inner, failingSummarizer, 5, 2);

    const history = makeHistory(6);
    for (const msg of history) {
      await inner.saveChatMessage("u", "s", "a", msg);
    }

    await expect(storage.fetchChat("u", "s", "a")).rejects.toThrow("summarizer failed");
  });

  test("different agentId slots are tracked independently", async () => {
    const summarizer = jest.fn(identitySummarizer);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    // agent1: above trigger
    const history1 = makeHistory(6);
    for (const msg of history1) {
      await inner.saveChatMessage("u", "s", "agent1", msg);
    }

    // agent2: below trigger
    const history2 = makeHistory(3);
    for (const msg of history2) {
      await inner.saveChatMessage("u", "s", "agent2", msg);
    }

    await storage.fetchChat("u", "s", "agent1");
    await storage.fetchChat("u", "s", "agent2");

    // Summarizer called only for agent1
    expect(summarizer).toHaveBeenCalledTimes(1);
  });
});
