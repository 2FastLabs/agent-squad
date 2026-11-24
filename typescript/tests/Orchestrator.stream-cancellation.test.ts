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
});
