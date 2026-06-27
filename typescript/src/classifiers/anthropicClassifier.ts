import {
  ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4, ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET,
  ConversationMessage,
  ParticipantRole,
} from "../types";
import { isClassifierToolInput } from "../utils/helpers";
import { Logger } from "../utils/logger";
import { Classifier, ClassifierCallbacks, ClassifierResult } from "./classifier";
import { Anthropic } from "@anthropic-ai/sdk";

export interface AnthropicClassifierOptions {
  // Optional: The ID of the Anthropic model to use for classification
  // If not provided, a default model may be used
  modelId?: string;

  // Optional: Configuration for the inference process
  inferenceConfig?: {
    // Maximum number of tokens to generate in the response
    maxTokens?: number;

    // Controls randomness in output generation
    // Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic
    temperature?: number;

    // Controls diversity of output via nucleus sampling
    // 1.0 considers all tokens, lower values (e.g., 0.9) consider only the most probable tokens
    topP?: number;

    // Array of sequences that will stop the model from generating further tokens when encountered
    stopSequences?: string[];
  };

  // The API key for authenticating with Anthropic's services
  apiKey: string;

  callbacks?: ClassifierCallbacks
}

export class AnthropicClassifier extends Classifier {
  private client: Anthropic;
  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };

  private tools: Anthropic.Tool[] = [
    {
      name: 'analyzePrompt',
      description: 'Analyze the user input and provide structured output',
      input_schema: {
        type: 'object',
        properties: {
          userinput: {
            type: 'string',
            description: 'The original user input',
          },
          selected_agent: {
            type: 'string',
            description: 'The name of the selected agent',
          },
          confidence: {
            type: 'number',
            description: 'Confidence level between 0 and 1',
          },
        },
        required: ['userinput', 'selected_agent', 'confidence'],
      },
    },
  ];

  protected callbacks: ClassifierCallbacks;


  constructor(options: AnthropicClassifierOptions) {
    super();

    if (!options.apiKey) {
      throw new Error("Anthropic API key is required");
    }
    this.client = new Anthropic({ apiKey: options.apiKey });
    this.modelId = options.modelId || ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4;
    // Set default value for max_tokens if not provided
    const defaultMaxTokens = 1000; // You can adjust this default value as needed
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? defaultMaxTokens,
      temperature: options.inferenceConfig?.temperature,
      stopSequences: options.inferenceConfig?.stopSequences,
    };

    this.callbacks = options.callbacks ?? new ClassifierCallbacks();

}

/* eslint-disable @typescript-eslint/no-unused-vars */
async processRequest(
    inputText: string,
    chatHistory: ConversationMessage[]
  ): Promise<ClassifierResult> {
    const userMessage: Anthropic.MessageParam = {
      role: ParticipantRole.USER,
      content: inputText,
    };

    try {
      const callParams: any = {
        model: this.modelId,
        max_tokens: this.inferenceConfig.maxTokens,
        messages: [userMessage],
        system: this.systemPrompt,
        temperature: this.inferenceConfig.temperature,
        tools: this.tools
      };
      // Only pass top_p if explicitly set — newer Anthropic models reject both temperature and top_p
      if (this.inferenceConfig.topP !== undefined) {
        callParams.top_p = this.inferenceConfig.topP;
      }
      const response = await this.client.messages.create(callParams);

      const toolUse = response.content.find(
        (content): content is Anthropic.ToolUseBlock => content.type === "tool_use"
      );

      if (!toolUse) {
        throw new Error("No tool use found in the response");
      }

      if (!isClassifierToolInput(toolUse.input)) {
        throw new Error("Tool input does not match expected structure");
      }


      // Create and return IntentClassifierResult
      const intentClassifierResult: ClassifierResult = {
        selectedAgent: this.getAgentById(toolUse.input.selected_agent),
        confidence: parseFloat(toolUse.input.confidence),
      };
      return intentClassifierResult;

    } catch (error) {
      Logger.logger.error("Error processing request:", error);
      // Instead of returning a default result, we'll throw the error
      throw error;
    }
  }


}
