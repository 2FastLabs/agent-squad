import { Agent, AgentCallbacks, AgentOptions } from "./agent";
import {
  ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4, ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET,
  ConversationMessage,
  ParticipantRole,
  TemplateVariables,
} from "../types";
import { Retriever } from "../retrievers/retriever";
import { Logger } from "../utils/logger";
import { Anthropic } from "@anthropic-ai/sdk";
import { AgentToolResult, AgentTools } from "../utils/tool";
import { isConversationMessage } from "../utils/helpers";

export interface AnthropicAgentOptions extends AgentOptions {
  modelId?: string;
  streaming?: boolean;
  toolConfig?: {
    tool: AgentTools | Anthropic.Tool[];
    useToolHandler: (response: any, conversation: any[]) => any;
    toolMaxRecursions?: number;
  };
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

  thinking?: {
    type: string;
    budget_tokens: number;
  };

  retriever?: Retriever;
  customSystemPrompt?: {
    template: string;
    variables?: TemplateVariables;
  };
  callbacks?: AgentCallbacks;
}

type WithApiKey = {
  apiKey: string;
  client?: never;
};

type WithClient = {
  client: Anthropic;
  apiKey?: never;
};

export type AnthropicAgentOptionsWithAuth = AnthropicAgentOptions &
  (WithApiKey | WithClient);

export class AnthropicAgent extends Agent {
  private client: Anthropic;
  protected streaming: boolean;
  private modelId: string;
  protected customSystemPrompt?: string;
  protected inferenceConfig: {
    maxTokens: number;
    temperature: number;
    topP: number;
    stopSequences: string[];
  };

  protected thinking?: {
    type: string;
    budget_tokens: number;
  };

  protected retriever?: Retriever;

  public toolConfig?: {
    tool: AgentTools | Anthropic.Tool[];
    useToolHandler?: (response: any, conversation: any[]) => any;
    toolMaxRecursions?: number;
  };

  private promptTemplate: string;
  private systemPrompt: string;
  private customVariables: TemplateVariables;
  private defaultMaxRecursions: number = 20;

  protected callbacks?: AgentCallbacks;

  constructor(options: AnthropicAgentOptionsWithAuth) {
    super(options);

    if (!options.apiKey && !options.client) {
      throw new Error("Anthropic API key or Anthropic client is required");
    }
    if (options.client) {
      this.client = options.client;
    } else {
      if (!options.apiKey) throw new Error("Anthropic API key is required");
      this.client = new Anthropic({ apiKey: options.apiKey });
    }

    this.systemPrompt = "";
    this.customVariables = {};

    this.streaming = options.streaming ?? false;

    this.modelId = options.modelId || ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4;

    this.thinking = options.thinking ?? null;

    const defaultMaxTokens = 1000;
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? defaultMaxTokens,
      temperature: options.inferenceConfig?.temperature ?? 0.1,
      stopSequences: options.inferenceConfig?.stopSequences ?? [],
    };
    // Only set topP if explicitly provided — newer Anthropic models reject both temperature and topP
    if (options.inferenceConfig?.topP !== undefined) {
      this.inferenceConfig.topP = options.inferenceConfig.topP;
    }

    this.retriever = options.retriever;

    this.toolConfig = options.toolConfig;

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
  private isAgentTools(
    tool: AgentTools | Anthropic.Tool[]
  ): tool is AgentTools {
    return tool instanceof AgentTools;
  }

  /**
   * Transforms the tools into a format compatible with Anthropic's Claude format.
   * This method maps each tool to an object containing its name, description, and input schema.
   *
   * @param tools - The Tools object containing an array of tools to be formatted.
   * @returns An array of tools in Claude's expected format.
   */
  private formatTools(tools: AgentTools): any[] {
    return tools.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: {
        type: "object",
        properties: tool.properties,
        required: tool.required,
      },
    }));
  }

  /**
   * Formats tool results into Anthropic's expected format
   * @param toolResults - Results from tool execution
   * @returns Formatted message in Anthropic's format
   */
  private formatToolResults(
    toolResults: AgentToolResult[] | any
  ): ConversationMessage {
    if (isConversationMessage(toolResults)) {
      return toolResults;
    }

    const result = {
      role: ParticipantRole.USER,
      content: toolResults.map((item: AgentToolResult) => ({
        type: "tool_result",
        tool_use_id: item.toolUseId,
        content: [{ type: "text", text: item.content }],
      })),
    };
    return result as ConversationMessage;
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
    // For Anthropic, the ID is under id, not toolUseId
    return toolUseBlock.id;
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
    const result = block.type === "tool_use" ? block : null;
    return result;
  }

  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    _additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    // Format messages to Anthropic's format
    const messages: Anthropic.MessageParam[] = chatHistory.map((message) => ({
      role:
        message.role === ParticipantRole.USER
          ? ParticipantRole.USER
          : ParticipantRole.ASSISTANT,
      content: message.content![0]["text"] || "", // Fallback to empty string if content is undefined
    }));
    messages.push({ role: ParticipantRole.USER, content: inputText });

    this.updateSystemPrompt();

    let systemPrompt = this.systemPrompt;

    // Update the system prompt with the latest history, agent descriptions, and custom variables
    if (this.retriever) {
      // retrieve from Vector store and combined results as a string into the prompt
      const response =
        await this.retriever.retrieveAndCombineResults(inputText);
      const contextPrompt =
        "\nHere is the context to use to answer the user's question:\n" +
        response;
      systemPrompt = systemPrompt + contextPrompt;
    }

    try {
      if (this.streaming) {
        return this.handleStreamingResponse(messages, systemPrompt);
      } else {
        let finalMessage: string = "";
        let thinkingMessage: string = "";
        let toolUse = false;
        let recursions =
          this.toolConfig?.toolMaxRecursions || this.defaultMaxRecursions;
        do {
          // Call Anthropic
          const llmInput: any = {
            model: this.modelId,
            max_tokens: this.inferenceConfig.maxTokens,
            messages: messages,
            system: systemPrompt,
            temperature: this.inferenceConfig.temperature,
            thinking: this.thinking,
            ...(this.toolConfig && {
              tools:
                this.toolConfig.tool instanceof AgentTools
                  ? this.formatTools(this.toolConfig.tool)
                  : this.toolConfig.tool,
            }),
          };
          // Only pass top_p if explicitly set — newer Anthropic models reject both temperature and top_p
          if (this.inferenceConfig.topP !== undefined) {
            llmInput.top_p = this.inferenceConfig.topP;
          }
          const response = await this.handleSingleResponse(llmInput);

          const toolUseBlocks = response.content.filter<Anthropic.ToolUseBlock>(
            (content) => content.type === "tool_use"
          );

          if (toolUseBlocks.length > 0) {
            // Append current response to the conversation
            messages.push({
              role: ParticipantRole.ASSISTANT,
              content: response.content,
            });

            const tools = this.toolConfig.tool;
            const toolHandler =
              this.toolConfig.useToolHandler ??
              (async (response, _conversationHistory) => {
                if (this.isAgentTools(tools)) {
                  return tools.toolHandler(
                    response,
                    this.getToolUseBlock.bind(this),
                    this.getToolName.bind(this),
                    this.getToolId.bind(this),
                    this.getInputData.bind(this)
                  );
                }
                // Legacy Tool[] requires an explicit useToolHandler
                throw new Error(
                  "toolConfig.useToolHandler is required when using Tool[] instead of AgentTools"
                );
              });

            const toolResponse = await toolHandler(response, messages);
            const formattedResponse = this.formatToolResults(toolResponse);

            // Add the formatted response to messages
            messages.push(formattedResponse);
            toolUse = true;
          } else {
            const textContent = response.content.find(
              (content): content is Anthropic.TextBlock =>
                content.type === "text"
            );
            finalMessage = textContent?.text || "";
            const thinkingContent = response.content.find(
              (content): content is Anthropic.ThinkingBlock =>
                content.type === "thinking"
            );

            thinkingMessage = thinkingContent?.thinking || "";
          }

          if (response.stop_reason === "end_turn") {
            toolUse = false;
          }

          recursions--;
        } while (toolUse && recursions > 0);

        return {
          role: ParticipantRole.ASSISTANT,
          content: [{ text: finalMessage, thinking: thinkingMessage }],
        };
      }
    } catch (error) {
      Logger.logger.error("Error processing request:", error);
      // Instead of returning a default result, we'll throw the error
      throw error;
    }
  }

  protected async handleSingleResponse(input: any): Promise<Anthropic.Message> {
    try {
      const response = await this.client.messages.create(input);
      return response as Anthropic.Message;
    } catch (error) {
      Logger.logger.error("Error invoking Anthropic:", error);
      throw error;
    }
  }

  private async *handleStreamingResponse(
    messages: any[],
    prompt: any
  ): AsyncIterable<string> {
    let toolUse = false;
    let recursions = this.toolConfig?.toolMaxRecursions || 5;

    do {
      const streamConfig: any = {
        model: this.modelId,
        max_tokens: this.inferenceConfig.maxTokens,
        messages: messages,
        system: prompt,
        temperature: this.inferenceConfig.temperature,
        thinking: {
          type: this.thinking?.type === "enabled" ? "enabled" : "disabled",
          budget_tokens: this.thinking?.budget_tokens
        },
        ...(this.toolConfig && {
          tools:
            this.toolConfig.tool instanceof AgentTools
              ? this.formatTools(this.toolConfig.tool)
              : this.toolConfig.tool,
        }),
      };
      // Only pass top_p if explicitly set — newer Anthropic models reject both temperature and top_p
      if (this.inferenceConfig.topP !== undefined) {
        streamConfig.top_p = this.inferenceConfig.topP;
      }
      const stream = await this.client.messages.stream(streamConfig);
      });

      let toolBlock: Anthropic.ToolUseBlock = {
        id: "",
        input: {},
        name: "",
        type: "tool_use",
      };
      let inputString = "";

      for await (const event of stream) {
        if (
          event.type === "content_block_delta" &&
          event.delta.type === "text_delta"
        ) {
          yield event.delta.text;
        } else if (
          event.type === "content_block_delta" &&
          event.delta.type === "thinking_delta"
        ) {
          yield JSON.stringify({
            thinking: true,
            content: event.delta.thinking,
          });
        } else if (
          event.type === "content_block_start" &&
          event.content_block.type === "tool_use"
        ) {
          if (!this.toolConfig?.tool) {
            throw new Error("No tools available for tool use");
          }
          toolBlock = event.content_block;
        } else if (
          event.type === "content_block_delta" &&
          event.delta.type === "input_json_delta"
        ) {
          inputString += event.delta.partial_json;
        } else if (event.type === "message_delta") {
          if (event.delta.stop_reason === "tool_use") {
            if (toolBlock && inputString) {
              toolBlock.input = JSON.parse(inputString);
              const message = { role: "assistant", content: [toolBlock] };
              messages.push(message);
              const tools = this.toolConfig.tool;
              const toolHandler =
                this.toolConfig.useToolHandler ??
                (async (response, _conversationHistory) => {
                  if (this.isAgentTools(tools)) {
                    return tools.toolHandler(
                      response,
                      this.getToolUseBlock.bind(this),
                      this.getToolName.bind(this),
                      this.getToolId.bind(this),
                      this.getInputData.bind(this)
                    );
                  }
                  // Legacy Tool[] requires an explicit useToolHandler
                  throw new Error(
                    "toolConfig.useToolHandler is required when using Tool[] instead of AgentTools"
                  );
                });

              const toolResponse = await toolHandler(message, messages);
              const formattedResponse = this.formatToolResults(toolResponse);

              // Add the formatted response to messages
              messages.push(formattedResponse);
              toolUse = true;
            }
          } else {
            toolUse = false;
          }
        }
      }
    } while (toolUse && --recursions > 0);
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
