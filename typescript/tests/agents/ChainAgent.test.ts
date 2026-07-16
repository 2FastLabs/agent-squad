import { ChainAgent, ChainAgentOptions } from '../../src/agents/chainAgent';
import { Agent, AgentOptions } from '../../src/agents/agent';
import { ConversationMessage, ParticipantRole } from '../../src/types';

// Minimal concrete Agent subclass for testing
class MockAgent extends Agent {
  private response: ConversationMessage | AsyncIterable<any>;

  constructor(
    options: AgentOptions,
    response: ConversationMessage | AsyncIterable<any>
  ) {
    super(options);
    this.response = response;
  }

  async processRequest(): Promise<ConversationMessage | AsyncIterable<any>> {
    return this.response;
  }
}

function makeMessage(text: string): ConversationMessage {
  return { role: ParticipantRole.ASSISTANT, content: [{ text }] };
}

function makeAgent(name: string, response: ConversationMessage | AsyncIterable<any>): MockAgent {
  return new MockAgent({ name, description: `${name} description` }, response);
}

describe('ChainAgent', () => {
  const defaultOptions: ChainAgentOptions = {
    name: 'TestChain',
    description: 'Test chain agent',
    agents: [makeAgent('Agent1', makeMessage('step one output'))],
  };

  describe('constructor', () => {
    it('throws when agents array is empty', () => {
      expect(() => new ChainAgent({ ...defaultOptions, agents: [] })).toThrow(
        'ChainAgent requires at least one agent in the chain.'
      );
    });

    it('initialises with a single agent', () => {
      const chain = new ChainAgent(defaultOptions);
      expect(chain.agents).toHaveLength(1);
    });

    it('uses custom defaultOutput when provided', () => {
      const chain = new ChainAgent({ ...defaultOptions, defaultOutput: 'custom fallback' });
      expect(chain).toBeDefined();
    });
  });

  describe('processRequest', () => {
    const userId = 'user1';
    const sessionId = 'session1';
    const chatHistory: ConversationMessage[] = [];

    it('returns the output of a single-agent chain', async () => {
      const chain = new ChainAgent(defaultOptions);
      const result = await chain.processRequest('hello', userId, sessionId, chatHistory) as ConversationMessage;

      expect(result.role).toBe(ParticipantRole.ASSISTANT);
      expect(result.content[0]).toHaveProperty('text', 'step one output');
    });

    it('pipes output of each agent as input to the next', async () => {
      const spy1 = jest.fn().mockResolvedValue(makeMessage('output of agent 1'));
      const spy2 = jest.fn().mockResolvedValue(makeMessage('output of agent 2'));

      const agent1 = new MockAgent({ name: 'A1', description: 'd' }, makeMessage(''));
      const agent2 = new MockAgent({ name: 'A2', description: 'd' }, makeMessage(''));
      agent1.processRequest = spy1 as any;
      agent2.processRequest = spy2 as any;

      const chain = new ChainAgent({ ...defaultOptions, agents: [agent1, agent2] });
      await chain.processRequest('initial input', userId, sessionId, chatHistory);

      expect(spy1).toHaveBeenCalledWith('initial input', userId, sessionId, chatHistory, undefined);
      expect(spy2).toHaveBeenCalledWith('output of agent 1', userId, sessionId, chatHistory, undefined);
    });

    it('passes additionalParams through to each agent', async () => {
      const spy = jest.fn().mockResolvedValue(makeMessage('ok'));
      const agent = new MockAgent({ name: 'A', description: 'd' }, makeMessage(''));
      agent.processRequest = spy as any;

      const chain = new ChainAgent({ ...defaultOptions, agents: [agent] });
      const params = { key: 'val' };
      await chain.processRequest('input', userId, sessionId, chatHistory, params);

      expect(spy).toHaveBeenCalledWith('input', userId, sessionId, chatHistory, params);
    });

    it('returns default response when an intermediate agent returns no text content', async () => {
      const emptyAgent = makeAgent('Empty', { role: ParticipantRole.ASSISTANT, content: [] });
      const secondAgent = makeAgent('Second', makeMessage('should not reach'));

      const chain = new ChainAgent({ ...defaultOptions, agents: [emptyAgent, secondAgent] });
      const result = await chain.processRequest('input', userId, sessionId, chatHistory) as ConversationMessage;

      expect(result.content[0]).toHaveProperty('text', 'No output generated from the chain.');
    });

    it('throws when an agent in the chain errors', async () => {
      const errorAgent = new MockAgent({ name: 'ErrorAgent', description: 'd' }, makeMessage(''));
      errorAgent.processRequest = jest.fn().mockRejectedValue(new Error('downstream failure')) as any;

      const chain = new ChainAgent({ ...defaultOptions, agents: [errorAgent] });
      await expect(chain.processRequest('input', userId, sessionId, chatHistory))
        .rejects.toMatch(/Error processing request with agent ErrorAgent/);
    });

    it('allows the last agent to return a streaming response', async () => {
      async function* streamGen() { yield 'chunk'; }
      const streamingAgent = makeAgent('Streamer', streamGen());

      const chain = new ChainAgent({ ...defaultOptions, agents: [streamingAgent] });
      const result = await chain.processRequest('input', userId, sessionId, chatHistory);

      expect(typeof (result as any)[Symbol.asyncIterator]).toBe('function');
    });

    it('returns default response when a non-last agent returns a streaming response', async () => {
      async function* streamGen() { yield 'chunk'; }
      const streamingAgent = makeAgent('Streamer', streamGen());
      const nextAgent = makeAgent('Next', makeMessage('next'));

      const chain = new ChainAgent({ ...defaultOptions, agents: [streamingAgent, nextAgent] });
      const result = await chain.processRequest('input', userId, sessionId, chatHistory) as ConversationMessage;

      expect(result.content[0]).toHaveProperty('text', 'No output generated from the chain.');
    });
  });
});
