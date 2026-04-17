import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  Tool,
} from "@aws-sdk/client-bedrock-runtime";
import { Agent, AgentCallbacks, AgentOptions } from "./agent";
import {
  BEDROCK_MODEL_ID_CLAUDE_3_HAIKU,
  ConversationMessage,
  ParticipantRole,
  TemplateVariables,
} from "../types";
import { Retriever } from "../retrievers/retriever";
import { Logger } from "../utils/logger";
import { AgentToolResult, AgentTools } from "../utils/tool";
import { isConversationMessage } from "../utils/helpers";
import { addUserAgentMiddleware } from "../common/src/awsSdkUtils";

export interface BedrockLLMAgentOptions extends AgentOptions {
  modelId?: string;
  region?: string;
  streaming?: boolean;
  inferenceConfig?: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  guardrailConfig?: {
    guardrailIdentifier: string;
    guardrailVersion: string;
  };
  reasoningConfig?: {
    thinking: {
      type: string;
      budget_tokens: number;
    };
  };
  retriever?: Retriever;
  toolConfig?: {
    tool: AgentTools | Tool[];
    useToolHandler: (response: any, conversation: ConversationMessage[]) => any;
    toolMaxRecursions?: number;
  };
  customSystemPrompt?: {
    template: string;
    variables?: TemplateVariables;
  };
  client?: BedrockRuntimeClient;
  callbacks?: AgentCallbacks;
}

/**
 * BedrockAgent class represents an agent that uses Amazon Bedrock for natural language processing.
 * It extends the base Agent class and implements the processRequest method using Bedrock's API.
 */
export class BedrockLLMAgent extends Agent {
  /** AWS Bedrock Runtime Client for making API calls */
  protected client: BedrockRuntimeClient;

  protected customSystemPrompt?: string;

  protected streaming: boolean;

  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };

  /**
   * The ID of the model used by this agent.
   */
  protected modelId?: string;

  protected guardrailConfig?: {
    guardrailIdentifier: string;
    guardrailVersion: string;
  };

  protected reasoningConfig?: {
    thinking: {
      type: string;
      budget_tokens: number;
    };
  };

  protected retriever?: Retriever;

  public toolConfig?: {
    tool: AgentTools | Tool[];
    useToolHandler?: (
      response: any,
      conversation: ConversationMessage[]
    ) => any;
    toolMaxRecursions?: number;
  };

  private promptTemplate: string;
  private systemPrompt: string;
  private customVariables: TemplateVariables;
  private defaultMaxRecursions: number = 20;

  protected callbacks?: AgentCallbacks;

  /**
   * Constructs a new BedrockAgent instance.
   * @param options - Configuration options for the agent, inherited from AgentOptions.
   */
  constructor(options: BedrockLLMAgentOptions) {
    super(options);

    this.client = options.client
      ? options.client
      : options.region
        ? new BedrockRuntimeClient({ region: options.region })
        : new BedrockRuntimeClient();

    addUserAgentMiddleware(this.client, "bedrock-llm-agent");

    // Initialize the modelId
    this.modelId = options.modelId ?? BEDROCK_MODEL_ID_CLAUDE_3_HAIKU;

    this.streaming = options.streaming ?? false;

    this.inferenceConfig = options.inferenceConfig ?? {};

    this.guardrailConfig = options.guardrailConfig ?? null;

    this.reasoningConfig = options.reasoningConfig ?? null;

    this.retriever = options.retriever ?? null;

    this.toolConfig = options.toolConfig ?? null;

    this.callbacks = options.callbacks ?? new AgentCallbacks();

    this.promptTemplate = `You are a ${this.name}. ${this.description} Provide helpful and accurate information based on your expertise.
    You will engage in an open-ended conversation, providing helpful and accurate information based on your expertise.
    The conversation will proceed as follows:
    - The human may ask an initial question or provide a prompt on any topic.
    - You will provide a relevant and informative response.
    - The human may then follow up with additional questions or prompts related to your previous response, allowing for a multi-turn dialogue on that topic.
    - Or, the human may switch to a completely new and unrelated topic at any point.
    - You will seamlessly shift your focus to the new topic, providing thoughtful and coherent responses based on your broad knowledge base.
    Throughout the conversation, you should aim to:
    - Understand the context and intent behind each new question or prompt.
    - Provide substantive and well-reasoned responses that directly address the query.
    - Draw insights and connections from your extensive knowledge when appropriate.
    - Ask for clarification if any part of the question or prompt is ambiguous.
    - Maintain a consistent, respectful, and engaging tone tailored to the human's communication style.
    - Seamlessly transition between topics as the human introduces new subjects.`;

    if (options.customSystemPrompt) {
      this.setSystemPrompt(
        options.customSystemPrompt.template,
        options.customSystemPrompt.variables
      );
    }
  }

  /**
   * Type guard to check if the tool is an AgentTools instance
   */
  private isAgentTools(tool: AgentTools | Tool[]): tool is AgentTools {
    return tool instanceof AgentTools;
  }

  /**
   * Formats the tool results into a conversation message format.
   * This method converts an array of tool results into a format expected by the system.
   *
   * @param toolResults - An array of ToolResult objects that need to be formatted.
   * @returns A ConversationMessage object containing the formatted tool results.
   */
  private formatToolResults(
    toolResults: AgentToolResult[]
  ): ConversationMessage {
    if (isConversationMessage(toolResults)) {
      return toolResults as ConversationMessage;
    }

    return {
      role: ParticipantRole.USER,
      content: toolResults.map((item: any) => ({
        toolResult: {
          toolUseId: item.toolUseId,
          content: [{ text: item.content }],
        },
      })),
    } as ConversationMessage;
  }

  /**
   * Transforms the tools into a format compatible with the system's expected structure.
   * This method maps each tool to an object containing its name, description, and input schema.
   *
   * @param tools - The Tools object containing an array of tools to be formatted.
   * @returns An array of formatted tool specifications.
   */
  private formatTools(tools: AgentTools): any[] {
    return tools.tools.map((tool) => ({
      toolSpec: {
        name: tool.name,
        description: tool.description,
        inputSchema: {
          json: {
            type: "object",
            properties: tool.properties,
            required: tool.required,
          },
        },
      },
    }));
  }

  /**
   * Extracts the tool name from the tool use block.
   * This method retrieves the `name` field from the provided tool use block.
   *
   * @param toolUseBlock - The block containing tool use details, including a `name` field.
   * @returns The name of the tool from the provided block.
   */
  private getToolName(toolUseBlock: any): string {
    return toolUseBlock.name;
  }

  /**
   * Extracts the tool ID from the tool use block.
   * This method retrieves the `toolUseId` field from the provided tool use block.
   *
   * @param toolUseBlock - The block containing tool use details, including a `toolUseId` field.
   * @returns The tool ID from the provided block.
   */
  private getToolId(toolUseBlock: any): string {
    return toolUseBlock.toolUseId;
  }

  /**
   * Extracts the input data from the tool use block.
   * This method retrieves the `input` field from the provided tool use block.
   *
   * @param toolUseBlock - The block containing tool use details, including an `input` field.
   * @returns The input data associated with the tool use block.
   */
  private getInputData(toolUseBlock: any): any {
    return toolUseBlock.input;
  }

  /**
   * Retrieves the tool use block from the provided block.
   * This method checks if the block contains a `toolUse` field and returns it.
   *
   * @param block - The block from which the tool use block needs to be extracted.
   * @returns The tool use block if present, otherwise null.
   */
  private getToolUseBlock(block: any): any {
    return block.toolUse;
  }

  /**
   * Abstract method to process a request.
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
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    try {
      // Construct the user's message based on the provided inputText
      const userMessage: ConversationMessage = {
        role: ParticipantRole.USER,
        content: [{ text: `${inputText}` }],
      };

      // Combine the existing chat history with the user's message
      const conversation: ConversationMessage[] = [...chatHistory, userMessage];

      this.updateSystemPrompt();

      let systemPrompt = this.systemPrompt;

      // Update the system prompt with the latest history, agent descriptions, and custom variables
      if (this.retriever) {
        // retrieve from Vector store
        const response =
          await this.retriever.retrieveAndCombineResults(inputText);
        const contextPrompt =
          "\nHere is the context to use to answer the user's question:\n" +
          response;
        systemPrompt = systemPrompt + contextPrompt;
      }

      // Prepare the command to converse with the Bedrock API
      const converseCmd = {
        modelId: this.modelId,
        messages: conversation,
        system: [{ text: systemPrompt }],
        inferenceConfig: {
          maxTokens: this.inferenceConfig.maxTokens,
          temperature: this.inferenceConfig.temperature,
          stopSequences: this.inferenceConfig.stopSequences,
          ...(this.reasoningConfig?.thinking.type !== "enable" && {
            topP: this.inferenceConfig.topP,
          }),
        },
        ...(this.guardrailConfig && {
          guardrailConfig: this.guardrailConfig,
        }),
        ...(this.reasoningConfig && {
          additionalModelRequestFields: this.reasoningConfig,
        }),
        ...(this.toolConfig && {
          toolConfig: {
            tools:
              this.toolConfig.tool instanceof AgentTools
                ? this.formatTools(this.toolConfig.tool)
                : this.toolConfig.tool,
          },
        }),
      };

      if (this.streaming) {
        return this.handleStreamingResponse(converseCmd);
      } else {
        let continueWithTools = false;
        let finalMessage: ConversationMessage = {
          role: ParticipantRole.USER,
          content: [],
        };
        let maxRecursions =
          this.toolConfig?.toolMaxRecursions || this.defaultMaxRecursions;

        do {
          // send the conversation to Amazon Bedrock
          const bedrockResponse = await this.handleSingleResponse(converseCmd);

          // Append the model's response to the ongoing conversation
          conversation.push(bedrockResponse);
          // process model response
          if (
            bedrockResponse?.content?.some((content) => "toolUse" in content)
          ) {
            // forward everything to the tool use handler
            const tools = this.toolConfig.tool;

            const toolHandler =
              this.toolConfig.useToolHandler ??
              (async (response, conversationHistory) => {
                if (this.isAgentTools(tools)) {
                  return tools.toolHandler(
                    response,
                    this.getToolUseBlock.bind(this),
                    this.getToolName.bind(this),
                    this.getToolId.bind(this),
                    this.getInputData.bind(this)
                  );
                }
                // Only use legacy handler when it's not AgentTools
                return this.toolConfig.useToolHandler(
                  response,
                  conversationHistory
                );
              });

            const toolResponse = await toolHandler(
              bedrockResponse,
              conversation
            );

            const formattedResponse = this.formatToolResults(toolResponse);

            continueWithTools = true;
            converseCmd.messages.push(formattedResponse);
          } else {
            continueWithTools = false;
            finalMessage = bedrockResponse;
          }
          maxRecursions--;

          converseCmd.messages = conversation;
        } while (continueWithTools && maxRecursions > 0);

        // If maxRecursions exhausted during tool loop, finalMessage may be null or contain toolUse blocks
        if (!finalMessage || finalMessage.content?.some((c: any) => 'toolUse' in c)) {
          return {
            role: ParticipantRole.ASSISTANT,
            content: [{ text: "Maximum tool recursion limit reached without a final response." }]
          } as ConversationMessage;
        }

        return finalMessage;
      }
    } catch (error) {
      Logger.logger.error("Error processing request:", error.message);
      throw `Error processing request: ${error.message}`;
    }
  }

  protected async handleSingleResponse(
    input: any
  ): Promise<ConversationMessage> {
    try {
      const command = new ConverseCommand(input);

      const response = await this.client.send(command);
      if (!response.output) {
        throw new Error("No output received from Bedrock model");
      }
      return response.output.message as ConversationMessage;
    } catch (error) {
      Logger.logger.error("Error invoking Bedrock model:", error.message);
      throw `Error invoking Bedrock model: ${error.message}`;
    }
  }

  private async *handleStreamingResponse(input: any): AsyncIterable<string | any> {
    let toolBlock: any = { toolUseId: "", input: {}, name: "" };
    let inputString = "";
    let toolUse = false;
    let recursions = this.toolConfig?.toolMaxRecursions || this.defaultMaxRecursions;
    try {
      do {
        const command = new ConverseStreamCommand(input);
        const response = await this.client.send(command);
        let reasoningContent = null;
        let accumulatedThinking = '';
        if (!response.stream) {
          throw new Error("No stream received from Bedrock model");
        }
        for await (const chunk of response.stream) {
          if (
            chunk.contentBlockDelta &&
            chunk.contentBlockDelta.delta &&
            chunk.contentBlockDelta.delta.text
          ) {
            yield chunk.contentBlockDelta.delta.text;
          } else if (chunk.contentBlockDelta?.delta?.reasoningContent?.text) {
            const thinking = {
              thinking: true,
              content: chunk.contentBlockDelta.delta.reasoningContent.text,
            };
            accumulatedThinking += chunk.contentBlockDelta.delta.reasoningContent.text;
            yield thinking;
          }
          else if (chunk.contentBlockDelta?.delta?.reasoningContent?.signature) {
            reasoningContent = {
              reasoningText:
              {
                text: accumulatedThinking,
                signature:chunk.contentBlockDelta.delta.reasoningContent.signature
              }
            }
          }
          else if (chunk.contentBlockStart?.start?.toolUse) {
            toolBlock = chunk.contentBlockStart?.start?.toolUse;
          } else if (chunk.contentBlockDelta?.delta?.toolUse) {
            inputString += chunk.contentBlockDelta.delta.toolUse.input;
          } else if (chunk.messageStop?.stopReason === "tool_use") {
            toolBlock.input = JSON.parse(inputString);
            let message = {
                role: ParticipantRole.ASSISTANT,
                content: [],
              };
            if (reasoningContent){
              message.content.push({reasoningContent:reasoningContent});
            }
            message.content.push({ toolUse: toolBlock })
            input.messages.push(message);

            const tools = this.toolConfig.tool;
            const toolHandler =
              this.toolConfig.useToolHandler ??
              (async (response, conversationHistory) => {
                if (this.isAgentTools(tools)) {
                  return tools.toolHandler(
                    response,
                    this.getToolUseBlock.bind(this),
                    this.getToolName.bind(this),
                    this.getToolId.bind(this),
                    this.getInputData.bind(this)
                  );
                }
                // Only use legacy handler when it's not AgentTools
                return this.toolConfig.useToolHandler(
                  response,
                  conversationHistory
                );
              });

            const toolResponse = await toolHandler(message, input.messages);
            const formattedResponse = this.formatToolResults(toolResponse);

            input.messages.push(formattedResponse);
            toolUse = true;
          } else if (chunk.messageStop?.stopReason === "end_turn") {
            toolUse = false;
          }
        }
      } while (toolUse && --recursions > 0);
    } catch (error) {
      Logger.logger.error(
        "Error getting stream from Bedrock model:",
        error.message
      );
      throw `Error getting stream from Bedrock model: ${error.message}`;
    }
  }

  setSystemPrompt(template?: string, variables?: TemplateVariables): void {
    if (template) {
      this.promptTemplate = template;
    }

    if (variables) {
      this.customVariables = variables;
    }

    this.updateSystemPrompt();
  }

  private updateSystemPrompt(): void {
    const allVariables: TemplateVariables = {
      ...this.customVariables,
    };

    this.systemPrompt = this.replaceplaceholders(
      this.promptTemplate,
      allVariables
    );

    //console.log("*** systemPrompt="+this.systemPrompt)
  }

  private replaceplaceholders(
    template: string,
    variables: TemplateVariables
  ): string {
    return template.replace(/{{(\w+)}}/g, (match, key) => {
      if (key in variables) {
        const value = variables[key];
        if (Array.isArray(value)) {
          return value.join("\n");
        }
        return value;
      }
      return match; // If no replacement found, leave the placeholder as is
    });
  }
}
