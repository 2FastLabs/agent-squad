
import { BedrockClassifier, BedrockClassifierOptions } from '../../src/classifiers/bedrockClassifier';
import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { ConversationMessage, BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5 } from "../../src/types/index";
import { MockAgent } from "../mock/mockAgent";
import { Logger } from "../../src/utils/logger";

const _logger = new Logger({}, console);

// Mock the BedrockRuntimeClient
jest.mock("@aws-sdk/client-bedrock-runtime");

describe('BedrockClassifier', () => {
  let classifier: BedrockClassifier;
  let mockSend: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockSend = jest.fn();
    (BedrockRuntimeClient as jest.Mock).mockImplementation(() => ({
      send: mockSend,
    }));

    

    const options: Partial<BedrockClassifierOptions> = {
      region: 'us-west-2',
      modelId: BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
      inferenceConfig: {
        maxTokens: 100,
        temperature: 0.7,
        topP: 0.9,
        stopSequences: ['stop'],
      },
    };
    classifier = new BedrockClassifier(options);
  });

  it('should initialize with default values when no options are provided', () => {
    const defaultClassifier = new BedrockClassifier();
    expect(defaultClassifier['region']).toBe(process.env.REGION);
    expect(defaultClassifier['modelId']).toBe(BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5);
    expect(defaultClassifier['inferenceConfig']).toEqual({});
  });

  it('should initialize with provided options', () => {
    expect(classifier['region']).toBe('us-west-2');
    expect(classifier['modelId']).toBe(BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5);
    expect(classifier['inferenceConfig']).toEqual({
      maxTokens: 100,
      temperature: 0.7,
      topP: 0.9,
      stopSequences: ['stop'],
    });
  });

  it('should process request successfully', async () => {
    const inputText = 'Hello, how can you help me?';
    const chatHistory: ConversationMessage[] = [];

    const mockResponse = {
      output: {
        message: {
          content: [
            {
              toolUse: {
                input: {
                  userinput: inputText,
                  selected_agent: 'MockAgent',
                  confidence: '0.95',
                },
              },
            },
          ],
        },
      },
    };

    mockSend.mockResolvedValue(mockResponse);

    classifier['getAgentById'] = jest.fn().mockReturnValue(new MockAgent({name:'agent', description:'agent description'}));

    const result = await classifier.processRequest(inputText, chatHistory);

    expect(result).toEqual({
      selectedAgent: expect.any(MockAgent),
      confidence: 0.95,
    });

    expect(mockSend).toHaveBeenCalledWith(expect.any(ConverseCommand));
    expect(classifier['getAgentById']).toHaveBeenCalledWith('MockAgent');
  });

  it('should throw an error when no output is received', async () => {
    mockSend.mockResolvedValue({});

    await expect(classifier.processRequest('input', [])).rejects.toThrow('No output received from Bedrock model');
  });

  it('should throw an error when no tool use is found', async () => {
    mockSend.mockResolvedValue({
      output: {
        message: {
          content: [{ text: 'Some response' }],
        },
      },
    });

    await expect(classifier.processRequest('input', [])).rejects.toThrow('No valid tool use found in the response');
  });

  it('should throw an error when tool input does not match expected structure', async () => {
    mockSend.mockResolvedValue({
      output: {
        message: {
          content: [
            {
              toolUse: {
                input: {
                  // Missing required fields
                  userinput: 'input',
                },
              },
            },
          ],
        },
      },
    });

    await expect(classifier.processRequest('input', [])).rejects.toThrow('Tool input does not match expected structure');
  });
});
