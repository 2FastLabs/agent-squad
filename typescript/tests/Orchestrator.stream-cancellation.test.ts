import { once } from 'events';

import { Agent } from '../src/agents/agent';
import { AgentSquad } from '../src/orchestrator';
import { Classifier } from '../src/classifiers/classifier';
import { InMemoryChatStorage } from '../src/storage/memoryChatStorage';

describe('AgentSquad streaming cancellation', () => {
  test('closes the provider iterator when the downstream transform is destroyed', async () => {
    let finalized = false;
    let nextCalls = 0;
    const stream = (async function* () {
      try {
        nextCalls++;
        yield 'first';
        nextCalls++;
        await new Promise((resolve) => setTimeout(resolve, 25));
        yield 'second';
        nextCalls++;
        yield 'third';
      } finally {
        finalized = true;
      }
    })();
    const agent = {
      id: 'streaming-agent',
      name: 'Streaming Agent',
      description: 'streams chunks',
      saveChat: false,
      processRequest: jest.fn().mockResolvedValue(stream),
    } as unknown as Agent;
    const classifier = {
      classify: jest.fn(),
      setAgents: jest.fn(),
    } as unknown as Classifier;
    const squad = new AgentSquad({
      classifier,
      storage: new InMemoryChatStorage(),
      config: { LOG_EXECUTION_TIMES: false },
    });

    const response = await squad.agentProcessRequest(
      'go',
      'user',
      'session',
      { selectedAgent: agent, confidence: 1 },
    );
    expect(response.streaming).toBe(true);
    const output = response.output as NodeJS.ReadableStream & { destroy: () => void };
    output.once('data', () => output.destroy());
    await once(output, 'close');
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(finalized).toBe(true);
    expect(nextCalls).toBeLessThan(3);
  });

  test("handles provider iterator cleanup failures without unhandled rejection", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (error: unknown): void => {
      unhandled.push(error);
    };
    process.on("unhandledRejection", onUnhandled);

    let returnCalls = 0;
    const iterator = {
      next: jest
        .fn()
        .mockResolvedValueOnce({ done: false, value: "first" })
        .mockImplementation(
          () =>
            new Promise((resolve) => {
              setTimeout(() => resolve({ done: true, value: undefined }), 15);
            })
        ),
      return: jest.fn(async () => {
        returnCalls++;
        throw new Error("provider cleanup failed");
      }),
    };
    const stream = {
      [Symbol.asyncIterator]: () => iterator,
    };
    const agent = {
      id: "streaming-agent",
      name: "Streaming Agent",
      description: "streams chunks",
      saveChat: false,
      processRequest: jest.fn().mockResolvedValue(stream),
    } as unknown as Agent;
    const classifier = {
      classify: jest.fn(),
      setAgents: jest.fn(),
    } as unknown as Classifier;
    const squad = new AgentSquad({
      classifier,
      storage: new InMemoryChatStorage(),
      config: { LOG_EXECUTION_TIMES: false },
    });

    try {
      const response = await squad.agentProcessRequest(
        "go",
        "user",
        "session",
        { selectedAgent: agent, confidence: 1 }
      );
      const output = response.output as NodeJS.ReadableStream & {
        destroy: (error?: Error) => void;
      };
      const closed = new Promise<void>((resolve) => {
        output.once("close", resolve);
      });
      output.once("error", () => undefined);
      output.once("data", () => output.destroy(new Error("client gone")));
      await closed;
      await new Promise((resolve) => setTimeout(resolve, 30));

      expect(returnCalls).toBe(1);
      expect(unhandled).toHaveLength(0);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });
});
