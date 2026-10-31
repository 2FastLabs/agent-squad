import {
  BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
  ConversationMessage,
  ParticipantRole,
} from "../types";
import {
  BedrockRuntimeClient,
  ContentBlock,
  ConverseCommand,
  ToolConfiguration
} from "@aws-sdk/client-bedrock-runtime";

import { Classifier, ClassifierCallbacks, ClassifierResult } from "./classifier";
import { isClassifierToolInput } from "../utils/helpers";
import { Logger } from "../utils/logger";

import { addUserAgentMiddleware } from '../common/src/awsSdkUtils'


export interface BedrockClassifierOptions {
  // Optional: The ID of the Bedrock model to use for classification
  // If not provided, a default model may be used
  modelId?: string;

  // Optional: The AWS region where the Bedrock model is used
  region?: string;

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
  callbacks?: ClassifierCallbacks;
}

/**
 * IntentClassifier class extends BedrockAgent to provide specialized functionality
 * for classifying user intents, selecting appropriate agents, and generating
 * structured response.
 */
export class BedrockClassifier extends Classifier{
  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  protected client: BedrockRuntimeClient;
  protected region: string;
  protected tools = [
    {
      toolSpec: {
        name: "analyzePrompt",
        description: "Analyze the user input and provide structured output",
        inputSchema: {
          json: {
            type: "object",
            properties: {
              userinput: {
                type: "string",
                description: "The original user input",
              },
              selected_agent: {
                type: "string",
                description: "The name of the selected agent",
              },
              confidence: {
                type: "number",
                description: "Confidence level between 0 and 1",
              },
            },
            required: ["userinput", "selected_agent", "confidence"],
          },
        },
      },
    },
  ];
  protected callbacks: ClassifierCallbacks;



  /**
   * Constructs a new IntentClassifier instance.
   * @param options - Configuration options for the agent, inherited from AgentOptions.
   */
  constructor(options: Partial<BedrockClassifierOptions> = {}) {
    super();

    // Initialize default values or use provided options
    this.region = options.region || process.env.REGION;
    this.client = new BedrockRuntimeClient({region:this.region});
    addUserAgentMiddleware(this.client, "bedrock-classifier");
    this.modelId = options.modelId || BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5;
    // Initialize inferenceConfig only if it's provided in options
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens,
      temperature: options.inferenceConfig?.temperature,
      topP: options.inferenceConfig?.topP,
      stopSequences: options.inferenceConfig?.stopSequences,
    };
    this.callbacks = options.callbacks ?? new ClassifierCallbacks();
  }

  /**
   * Method to process a request.
   * This method must be implemented by all concrete agent classes.
   *
   * @param inputText - The user input as a string.
   * @param chatHistory - An array of Message objects representing the conversation history.
   * @param additionalParams - Optional additional parameters as key-value pairs.
   * @returns A Promise that resolves to a Message object containing the agent's response.
   */
  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    chatHistory: ConversationMessage[]
  ): Promise<ClassifierResult> {
    // Construct the user's message based on the provided inputText
    const userMessage: ConversationMessage = {
      role: ParticipantRole.USER,
      content: [{ text: inputText }],
    };

    const toolConfig: ToolConfiguration = {
      tools: this.tools,
    };

    // ToolChoice is only supported by Anthropic Claude 3 models and by Mistral AI Mistral Large.
    // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
    if (this.modelId.includes("anthropic") || this.modelId.includes("mistral-large")) {
      toolConfig.toolChoice = {
          tool: {
              name: "analyzePrompt",
          },
      };
    }

    // Prepare the command to converse with the Bedrock API
    const converseCmd = {
      modelId: this.modelId,
      messages: [userMessage],
      system: [{ text: this.systemPrompt }],
      toolConfig: toolConfig,
      inferenceConfiguration: {
        maximumTokens: this.inferenceConfig.maxTokens,
        temperature: this.inferenceConfig.temperature,
        topP: this.inferenceConfig.topP,
        stopSequences: this.inferenceConfig.stopSequences,
      },
    };

    try {
      const command = new ConverseCommand(converseCmd);
      const response = await this.client.send(command);

      if (!response.output) {
        throw new Error("No output received from Bedrock model");
      }
      if (response.output.message.content) {
        const responseContentBlocks = response.output.message
          .content as ContentBlock[];

        for (const contentBlock of responseContentBlocks) {
          if ("toolUse" in contentBlock) {
            const toolUse = contentBlock.toolUse;
              if (!toolUse) {
                throw new Error("No tool use found in the response");
              }

              if (!isClassifierToolInput(toolUse.input)) {
                throw new Error("Tool input does not match expected structure");
              }

              const intentClassifierResult: ClassifierResult = {
                selectedAgent: this.getAgentById(toolUse.input.selected_agent),
                confidence: parseFloat(toolUse.input.confidence),
              };
              return intentClassifierResult;
          }
        }
      }

      throw new Error("No valid tool use found in the response");
    } catch (error) {
      Logger.logger.error("Error processing request:", error);
      // Instead of returning a default result, we'll throw the error
      throw error;
    }
  }


}