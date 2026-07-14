import { InMemoryChatStorage } from "../../src/storage/memoryChatStorage";
import { SummarizingChatStorage } from "../../src/storage/summarizingChatStorage";
import { ConversationMessage, ParticipantRole } from "../../src/types";

const user = (t: string): ConversationMessage => ({ role: ParticipantRole.USER, content: [{ text: t }] });
const assistant = (t: string): ConversationMessage => ({ role: ParticipantRole.ASSISTANT, content: [{ text: t }] });
const text = (m: ConversationMessage): string => (m.content as Array<{ text: string }>)[0].text;

const makeHistory = (pairs: number): ConversationMessage[] => {
  const msgs: ConversationMessage[] = [];
  for (let i = 0; i < pairs; i++) {
    msgs.push(user(`User ${i + 1}`));
    msgs.push(assistant(`Asst ${i + 1}`));
  }
  return msgs;
};

const seed = async (inner: InMemoryChatStorage, msgs: ConversationMessage[]) => {
  for (const msg of msgs) await inner.saveChatMessage("u", "s", "a", msg);
};

describe("SummarizingChatStorage", () => {

  // -------------------------------------------------------------------------
  // fetchChat — lazy buffer activation
  // -------------------------------------------------------------------------

  test("below trigger returns raw history", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(3));
    const summarizer = jest.fn(async (h: ConversationMessage[]) => h);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    expect(await storage.fetchChat("u", "s", "a")).toHaveLength(6);
    expect(summarizer).not.toHaveBeenCalled();
  });

  test("at boundary no summarization", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(5));
    const summarizer = jest.fn(async (h: ConversationMessage[]) => h);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    expect(await storage.fetchChat("u", "s", "a")).toHaveLength(10);
    expect(summarizer).not.toHaveBeenCalled();
  });

  test("above trigger calls summarizer on first fetch", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const summarizer = jest.fn(async (h: ConversationMessage[], k: number) => h.slice(-k * 2));
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    await storage.fetchChat("u", "s", "a");

    expect(summarizer).toHaveBeenCalledTimes(1);
    expect(summarizer).toHaveBeenCalledWith(expect.any(Array), 2);
  });

  test("fetch returns compressed result", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const storage = new SummarizingChatStorage(inner, jest.fn(async () => [user("Summary")]), 5, 2);

    const result = await storage.fetchChat("u", "s", "a");

    expect(result).toHaveLength(1);
    expect(text(result[0])).toBe("Summary");
  });

  test("subsequent fetch returns buffer without calling summarizer again", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const summarizer = jest.fn(async () => [user("Summary")]);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    await storage.fetchChat("u", "s", "a");
    const result = await storage.fetchChat("u", "s", "a");

    expect(summarizer).toHaveBeenCalledTimes(1);
    expect(text(result[0])).toBe("Summary");
  });

  // -------------------------------------------------------------------------
  // save — pure delegation before buffer active
  // -------------------------------------------------------------------------

  test("save before buffer active delegates to inner without calling summarizer", async () => {
    const inner = new InMemoryChatStorage();
    const summarizer = jest.fn();
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    await storage.saveChatMessage("u", "s", "a", user("Hello"));

    expect(await inner.fetchChat("u", "s", "a")).toHaveLength(1);
    expect(summarizer).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // save — buffer management after activation
  // -------------------------------------------------------------------------

  test("save after buffer active appends to buffer", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const storage = new SummarizingChatStorage(inner, jest.fn(async () => [user("Summary")]), 5, 2);

    await storage.fetchChat("u", "s", "a");
    await storage.saveChatMessage("u", "s", "a", user("New"));

    const result = await storage.fetchChat("u", "s", "a");
    expect(result).toHaveLength(2);
    expect(text(result[1])).toBe("New");
  });

  test("save triggers compression when buffer exceeds threshold", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));

    let callCount = 0;
    const summarizer = jest.fn(async () => [user(`Summary ${++callCount}`)]);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    await storage.fetchChat("u", "s", "a");
    expect(callCount).toBe(1);

    for (let i = 0; i < 11; i++) {
      await storage.saveChatMessage("u", "s", "a", i % 2 === 0 ? user(`m${i}`) : assistant(`m${i}`));
    }

    expect(callCount).toBe(2);
    expect(text((await storage.fetchChat("u", "s", "a"))[0])).toBe("Summary 2");
  });

  // -------------------------------------------------------------------------
  // fetchAllChats — never intercepted
  // -------------------------------------------------------------------------

  test("fetchAllChats returns raw history regardless of buffer", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const summarizer = jest.fn(async () => [user("Summary")]);
    const storage = new SummarizingChatStorage(inner, summarizer, 5, 2);

    await storage.fetchChat("u", "s", "a");
    const result = await storage.fetchAllChats("u", "s");

    expect(result).toHaveLength(12);
    expect(summarizer).toHaveBeenCalledTimes(1);
  });

  // -------------------------------------------------------------------------
  // Base storage integrity
  // -------------------------------------------------------------------------

  test("base storage always receives raw messages unmodified", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const storage = new SummarizingChatStorage(inner, jest.fn(async () => [user("Summary")]), 5, 2);

    await storage.fetchChat("u", "s", "a");
    await storage.saveChatMessage("u", "s", "a", user("New"));

    const raw = await inner.fetchChat("u", "s", "a");
    expect(raw).toHaveLength(13);
    expect(text(raw[12])).toBe("New");
  });

  // -------------------------------------------------------------------------
  // Error propagation
  // -------------------------------------------------------------------------

  test("summarizer error propagates from fetchChat", async () => {
    const inner = new InMemoryChatStorage();
    await seed(inner, makeHistory(6));
    const storage = new SummarizingChatStorage(
      inner,
      jest.fn(async () => { throw new Error("summarizer failed"); }),
      5, 2
    );

    await expect(storage.fetchChat("u", "s", "a")).rejects.toThrow("summarizer failed");
  });
});
