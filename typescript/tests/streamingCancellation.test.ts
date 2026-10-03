import { AgentSquad } from "../src/orchestrator";
import { Agent } from "../src/agents/agent";
import { Classifier, ClassifierResult } from "../src/classifiers/classifier";
import { ChatStorage } from "../src/storage/chatStorage";
import { ConversationMessage } from "../src/types";
import { AccumulatorTransform } from "../src/utils/helpers";

class TestClassifier extends Classifier {
  private selectedAgent: Agent | null = null;

  setSelectedAgent(agent: Agent): void {
    this.selectedAgent = agent;
  }

  async processRequest(
    _inputText: string,
    _chatHistory: ConversationMessage[],
  ): Promise<ClassifierResult> {
    return { selectedAgent: this.selectedAgent, confidence: 1 };
  }
}

class TestStorage extends ChatStorage {
  readonly savedMessages: ConversationMessage[] = [];

  async saveChatMessage(
    _userId: string,
    _sessionId: string,
    _agentId: string,
    message: ConversationMessage,
  ): Promise<ConversationMessage[]> {
    this.savedMessages.push(message);
    return this.savedMessages;
  }

  async fetchChat(): Promise<ConversationMessage[]> {
    return [];
  }

  async fetchAllChats(): Promise<ConversationMessage[]> {
    return [];
  }
}

function createAgent(
  processRequest: Agent["processRequest"],
  saveChat = true,
): Agent {
  return {
    id: "stream-agent",
    name: "Stream Agent",
    description: "A test streaming agent",
    saveChat,
    processRequest,
  } as Agent;
}

describe("streaming response lifecycle", () => {
  it("stops requesting chunks and closes the provider iterator when the output closes", async () => {
    let nextCalls = 0;
    let iteratorReturned = false;
    const pendingNext = new Promise<IteratorResult<string>>(() => undefined);
    const stream: AsyncIterable<string> = {
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          nextCalls += 1;
          if (nextCalls === 1) {
            return { done: false, value: "first" };
          }
          return pendingNext;
        },
        return: async () => {
          iteratorReturned = true;
          return { done: true, value: undefined };
        },
      }),
    };
    const agent = createAgent(async () => stream, false);
    const classifier = new TestClassifier();
    classifier.setSelectedAgent(agent);
    const orchestrator = new AgentSquad({
      classifier,
      storage: new TestStorage(),
    });
    orchestrator.addAgent(agent);

    const response = await orchestrator.routeRequest(
      "input",
      "user",
      "session",
    );
    const output = response.output as AccumulatorTransform;
    output.on("error", () => undefined);
    await new Promise<void>((resolve) => output.once("data", () => resolve()));
    output.destroy();

    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(iteratorReturned).toBe(true);
    expect(nextCalls).toBe(1);
  });

  it("saves a finite stream after delivering all chunks", async () => {
    const storage = new TestStorage();
    const agent = createAgent(async () =>
      (async function* () {
        yield "hello ";
        yield "world";
      })(),
    );
    const classifier = new TestClassifier();
    classifier.setSelectedAgent(agent);
    const orchestrator = new AgentSquad({ classifier, storage });
    orchestrator.addAgent(agent);

    const response = await orchestrator.routeRequest(
      "input",
      "user",
      "session",
    );
    const chunks: string[] = [];
    for await (const chunk of response.output as AccumulatorTransform) {
      chunks.push(chunk as string);
    }

    expect(chunks.join("")).toBe("hello world");
    expect(storage.savedMessages).toHaveLength(2);
    expect(storage.savedMessages[1].content[0].text).toBe("hello world");
  });

  it("rejects a stream that exceeds the configured byte limit", async () => {
    const storage = new TestStorage();
    const agent = createAgent(async () =>
      (async function* () {
        yield "12345";
        yield "6";
      })(),
    );
    const classifier = new TestClassifier();
    classifier.setSelectedAgent(agent);
    const orchestrator = new AgentSquad({
      classifier,
      storage,
      config: { MAX_STREAM_BYTES: 5 },
    });
    orchestrator.addAgent(agent);

    const response = await orchestrator.routeRequest(
      "input",
      "user",
      "session",
    );
    const output = response.output as AccumulatorTransform;
    const streamError = new Promise<Error>((resolve) =>
      output.once("error", resolve),
    );
    output.resume();

    await expect(streamError).resolves.toMatchObject({
      message: "Streaming response exceeded configured limits",
    });
    expect(storage.savedMessages).toHaveLength(0);
  });

  it("does not save an accumulated response that exceeds its text limit", async () => {
    const storage = new TestStorage();
    const agent = createAgent(async () =>
      (async function* () {
        yield "12345";
        yield "6";
      })(),
    );
    const classifier = new TestClassifier();
    classifier.setSelectedAgent(agent);
    const orchestrator = new AgentSquad({
      classifier,
      storage,
      config: {
        MAX_STREAM_BYTES: 100,
        MAX_ACCUMULATED_RESPONSE_BYTES: 5,
      },
    });
    orchestrator.addAgent(agent);

    const response = await orchestrator.routeRequest(
      "input",
      "user",
      "session",
    );
    const output = response.output as AccumulatorTransform;
    const streamError = new Promise<Error>((resolve) =>
      output.once("error", resolve),
    );
    output.resume();

    await expect(streamError).resolves.toMatchObject({
      message: "Maximum accumulated streaming response size exceeded",
    });
    expect(storage.savedMessages).toHaveLength(0);
  });

  it("forwards provider errors and closes the iterator", async () => {
    const storage = new TestStorage();
    const providerError = new Error("provider failed");
    const iterator = {
      next: jest.fn().mockRejectedValue(providerError),
      return: jest.fn().mockResolvedValue({ done: true, value: undefined }),
    };
    const stream: AsyncIterable<string> = {
      [Symbol.asyncIterator]: () => iterator,
    };
    const agent = createAgent(async () => stream);
    const classifier = new TestClassifier();
    classifier.setSelectedAgent(agent);
    const orchestrator = new AgentSquad({ classifier, storage });
    orchestrator.addAgent(agent);

    const response = await orchestrator.routeRequest(
      "input",
      "user",
      "session",
    );
    const output = response.output as AccumulatorTransform;
    const streamError = new Promise<Error>((resolve) =>
      output.once("error", resolve),
    );
    output.resume();

    await expect(streamError).resolves.toBe(providerError);
    expect(iterator.return).toHaveBeenCalledTimes(1);
    expect(storage.savedMessages).toHaveLength(0);
  });
});
