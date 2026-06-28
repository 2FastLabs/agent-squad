import { BedrockFlowsAgent, BedrockFlowsAgentOptions } from '../../src/agents/bedrockFlowsAgent';
import { BedrockAgentRuntimeClient, InvokeFlowCommand } from '@aws-sdk/client-bedrock-agent-runtime';
import { ConversationMessage, ParticipantRole } from '../../src/types';

jest.mock('@aws-sdk/client-bedrock-agent-runtime');
jest.mock('../../src/common/src/awsSdkUtils', () => ({
  addUserAgentMiddleware: jest.fn(),
}));

describe('BedrockFlowsAgent', () => {
  let mockClient: jest.Mocked<BedrockAgentRuntimeClient>;

  const defaultOptions: BedrockFlowsAgentOptions = {
    name: 'TestFlowsAgent',
    description: 'Test BedrockFlowsAgent',
    flowIdentifier: 'flow-123',
    flowAliasIdentifier: 'alias-456',
  };

  function makeResponseStream(document: string) {
    async function* gen() {
      yield { flowOutputEvent: { content: { document } } };
    }
    return { responseStream: gen() };
  }

  beforeEach(() => {
    jest.clearAllMocks();
    mockClient = { send: jest.fn(), use: jest.fn(), destroy: jest.fn() } as any;
    (BedrockAgentRuntimeClient as jest.Mock).mockImplementation(() => mockClient);
    (InvokeFlowCommand as unknown as jest.Mock).mockImplementation((p) => p);
  });

  describe('constructor', () => {
    it('creates a BedrockAgentRuntimeClient with no arguments when no region given', () => {
      new BedrockFlowsAgent(defaultOptions);
      expect(BedrockAgentRuntimeClient).toHaveBeenCalledWith();
    });

    it('creates a BedrockAgentRuntimeClient with specified region', () => {
      new BedrockFlowsAgent({ ...defaultOptions, region: 'eu-west-1' });
      expect(BedrockAgentRuntimeClient).toHaveBeenCalledWith({ region: 'eu-west-1' });
    });

    it('uses a pre-built client when bedrockAgentClient is provided', () => {
      const prebuilt = { send: jest.fn() } as any;
      const agent = new BedrockFlowsAgent({ ...defaultOptions, bedrockAgentClient: prebuilt });
      expect(agent).toBeDefined();
      expect(BedrockAgentRuntimeClient).not.toHaveBeenCalled();
    });

    it('defaults enableTrace to false', () => {
      const agent = new BedrockFlowsAgent(defaultOptions);
      expect((agent as any).enableTrace).toBe(false);
    });

    it('sets enableTrace when provided', () => {
      const agent = new BedrockFlowsAgent({ ...defaultOptions, enableTrace: true });
      expect((agent as any).enableTrace).toBe(true);
    });
  });

  describe('processRequest', () => {
    const userId = 'user1';
    const sessionId = 'session1';
    const chatHistory: ConversationMessage[] = [];

    it('invokes the flow and returns the decoded response', async () => {
      mockClient.send.mockResolvedValueOnce(makeResponseStream('Flow answer') as never);

      const agent = new BedrockFlowsAgent(defaultOptions);
      const result = await agent.processRequest('What is 2+2?', userId, sessionId, chatHistory);

      expect(mockClient.send).toHaveBeenCalledTimes(1);
      expect(result).toEqual<ConversationMessage>({
        role: ParticipantRole.ASSISTANT,
        content: [{ text: 'Flow answer' }],
      });
    });

    it('throws when the response stream is missing', async () => {
      mockClient.send.mockResolvedValueOnce({ responseStream: null } as never);

      const agent = new BedrockFlowsAgent(defaultOptions);
      await expect(agent.processRequest('input', userId, sessionId, chatHistory))
        .rejects.toThrow(/Error processing request with Bedrock/);
    });

    it('uses a custom flowInputEncoder when provided', async () => {
      const encoder = jest.fn().mockReturnValue('encoded input');
      mockClient.send.mockResolvedValueOnce(makeResponseStream('ok') as never);

      const agent = new BedrockFlowsAgent({ ...defaultOptions, flowInputEncoder: encoder });
      await agent.processRequest('hello', userId, sessionId, chatHistory);

      expect(encoder).toHaveBeenCalledWith(agent, 'hello', expect.objectContaining({ userId, sessionId }));
    });

    it('uses a custom flowOutputDecoder when provided', async () => {
      const decoded: ConversationMessage = { role: ParticipantRole.ASSISTANT, content: [{ text: 'custom' }] };
      const decoder = jest.fn().mockReturnValue(decoded);
      mockClient.send.mockResolvedValueOnce(makeResponseStream('raw') as never);

      const agent = new BedrockFlowsAgent({ ...defaultOptions, flowOutputDecoder: decoder });
      const result = await agent.processRequest('input', userId, sessionId, chatHistory);

      expect(decoder).toHaveBeenCalledWith(agent, 'raw');
      expect(result).toEqual(decoded);
    });

    it('wraps client errors in a descriptive error message', async () => {
      mockClient.send.mockRejectedValueOnce(new Error('Network failure') as never);

      const agent = new BedrockFlowsAgent(defaultOptions);
      await expect(agent.processRequest('input', userId, sessionId, chatHistory))
        .rejects.toThrow('Error processing request with Bedrock: Network failure');
    });
  });
});
