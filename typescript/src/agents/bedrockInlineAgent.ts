import {
    BedrockRuntimeClient,
    ConverseCommand,
  } from "@aws-sdk/client-bedrock-runtime";

  import { BedrockAgentRuntimeClient, InvokeInlineAgentCommand, AgentActionGroup, KnowledgeBase } from "@aws-sdk/client-bedrock-agent-runtime";
  import { Agent, AgentOptions } from "./agent";
  import {
    BEDROCK_MODEL_ID_CLAUDE_3_HAIKU,
    BEDROCK_MODEL_ID_CLAUDE_3_SONNET,
    ConversationMessage,
    ParticipantRole,
    TemplateVariables,
  } from "../types";

  export interface BedrockInlineAgentOptions extends AgentOptions {
    inferenceConfig?: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      stopSequences?: string[];
    };
    client?: BedrockRuntimeClient;
    bedrockAgentClient?: BedrockAgentRuntimeClient;
    modelId?: string;
    foundationModel?: string;
    region?: string;
    actionGroupsList?: AgentActionGroup[];
    knowledgeBases?: KnowledgeBase[];
    enableTrace?: boolean;
    customSystemPrompt?: {
      template?: string;
      variables?: TemplateVariables;
    };

  }

  export class BedrockInlineAgent extends Agent {
    /** Class-level constants */
    protected static readonly TOOL_INPUT_SCHEMA = {
      json: {
        type: "object",
        properties: {
          action_group_names: {
            type: "array",
            items: { type: "string" },
            description: "A string array of action group names needed to solve the customer request"
          },
          knowledge_bases: {
            type: "array",
            items: { type: "string" },
            description: "A string array of knowledge base names needed to solve the customer request"
          },
          description: {
            type: "string",
            description: "Description to instruct the agent how to solve the user request using available action groups"
          },
          user_request: {
            type: "string",
            description: "The initial user request"
          }
        },
        required: ["action_group_names", "description", "user_request"]
      }
    };

    protected static readonly KEYS_TO_REMOVE = [
      'actionGroupId',
      'actionGroupState',
      'agentId',
      'agentVersion'
    ];

    protected static readonly TOOL_NAME = "inline_agent_creation";

    /** Protected class members */
    protected client: BedrockRuntimeClient;
    protected bedrockAgentClient: BedrockAgentRuntimeClient;
    protected modelId: string;
    protected foundationModel: string;
    protected inferenceConfig: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      stopSequences?: string[];
    };
    protected actionGroupsList: AgentActionGroup[];
    protected knowledgeBases: KnowledgeBase[];
    protected enableTrace: boolean;
    protected inlineAgentTool: any[];
    protected toolConfig: {
      tool: any[];
      useToolHandler: (response: ConversationMessage, conversation: ConversationMessage[], sessionId: string) => Promise<ConversationMessage>;
      toolMaxRecursions: number;
    };

    private promptTemplate: string;
    private systemPrompt: string = '';
    private customVariables: TemplateVariables = {};

    constructor(options: BedrockInlineAgentOptions) {
      super(options);



      // Initialize clients
      this.client = options.client ?? (
        options.region
          ? new BedrockRuntimeClient({ region: options.region })
          : new BedrockRuntimeClient()
      );

      this.bedrockAgentClient = options.bedrockAgentClient ?? (
        options.region
          ? new BedrockAgentRuntimeClient({ region: options.region })
          : new BedrockAgentRuntimeClient()
      );

      // Set model IDs
      this.modelId = options.modelId ?? BEDROCK_MODEL_ID_CLAUDE_3_HAIKU;
      this.foundationModel = options.foundationModel ?? BEDROCK_MODEL_ID_CLAUDE_3_SONNET;

      this.enableTrace = options.enableTrace ?? false;

      // Set inference configuration
      this.inferenceConfig = options.inferenceConfig ?? {
        maxTokens: 1000,
        temperature: 0.0,
        topP: 0.9,
        stopSequences: []
      };

      // Store action groups and knowledge bases
      this.actionGroupsList = options.actionGroupsList ?? [];
      this.knowledgeBases = options.knowledgeBases ?? [];

      // Define inline agent tool configuration
      this.inlineAgentTool = [{
        toolSpec: {
          name: BedrockInlineAgent.TOOL_NAME,
          description: "Create an inline agent with a list of action groups and knowledge bases",
          inputSchema: BedrockInlineAgent.TOOL_INPUT_SCHEMA
        }
      }];

      // Configure tool usage
      this.toolConfig = {
        tool: this.inlineAgentTool,
        useToolHandler: this.inlineAgentToolHandler.bind(this),
        toolMaxRecursions: 1
      };

      // Set prompt template
      this.promptTemplate = `You are a ${this.name}.
      ${this.description}
      You will engage in an open-ended conversation,
      providing helpful and accurate information based on your expertise.
      The conversation will proceed as follows:
      - The human may ask an initial question or provide a prompt on any topic.
      - You will provide a relevant and informative response.
      - The human may then follow up with additional questions or prompts related to your previous
      response, allowing for a multi-turn dialogue on that topic.
      - Or, the human may switch to a completely new and unrelated topic at any point.
      - You will seamlessly shift your focus to the new topic, providing thoughtful and
      coherent responses based on your broad knowledge base.
      Throughout the conversation, you should aim to:
      - Understand the context and intent behind each new question or prompt.
      - Provide substantive and well-reasoned responses that directly address the query.
      - Draw insights and connections from your extensive knowledge when appropriate.
      - Ask for clarification if any part of the question or prompt is ambiguous.
      - Maintain a consistent, respectful, and engaging tone tailored
      to the human's communication style.
      - Seamlessly transition between topics as the human introduces new subjects.`;

      this.promptTemplate += "\n\nHere are the action groups that you can use to solve the customer request:\n";
      this.promptTemplate += "<action_groups>\n";

      for (const actionGroup of this.actionGroupsList) {
        this.promptTemplate += `Action Group Name: ${actionGroup.actionGroupName ?? ''}\n`;
        this.promptTemplate += `Action Group Description: ${actionGroup.description ?? ''}\n`;

      }

      this.promptTemplate += "</action_groups>\n";
      this.promptTemplate += "\n\nHere are the knowledge bases that you can use to solve the customer request:\n";
      this.promptTemplate += "<knowledge_bases>\n";

      for (const kb of this.knowledgeBases) {
        this.promptTemplate += `Knowledge Base ID: ${kb.knowledgeBaseId ?? ''}\n`;
        this.promptTemplate += `Knowledge Base Description: ${kb.description ?? ''}\n`;
      }

      this.promptTemplate += "</knowledge_bases>\n";


      if (options.customSystemPrompt) {
        this.setSystemPrompt(
          options.customSystemPrompt.template,
          options.customSystemPrompt.variables
        );
      }



    }

    private async inlineAgentToolHandler(
      response: ConversationMessage,
      conversation: ConversationMessage[],
      sessionId: string
    ): Promise<ConversationMessage> {
      const responseContentBlocks = response.content;

      if (!responseContentBlocks) {
        throw new Error("No content blocks in response");
      }

      for (const contentBlock of responseContentBlocks) {
        if ("toolUse" in contentBlock) {
          const toolUseBlock = contentBlock.toolUse;
          const toolUseName = toolUseBlock?.name;

          if (toolUseName === "inline_agent_creation") {
            // Get valid action group names from the tool use input
            const actionGroupNames = toolUseBlock.input?.action_group_names || [];
            const kbNames = toolUseBlock.input?.knowledge_bases || '';
            const description = toolUseBlock.input?.description || '';
            const userRequest = toolUseBlock.input?.user_request || '';


            this.logDebug("BedrockInlineAgent", 'Tool Handler Parameters', {
              userRequest,
              actionGroupNames,
              knowledgeBases: kbNames,
              description,
              sessionId
            });

            const actionGroups = this.actionGroupsList
              .filter(item => actionGroupNames.includes(item.actionGroupName)) // Keep only requested action groups
              .map(item => ({
                actionGroupName: item.actionGroupName,
                parentActionGroupSignature: item.parentActionGroupSignature,
                // Only include description if it's not a child action group
                ...(item.parentActionGroupSignature ? {} : { description: item.description })
              }));

            const kbs = kbNames && this.knowledgeBases.length
            ? this.knowledgeBases.filter(item => kbNames.includes(item.knowledgeBaseId))
            : [];

            this.logDebug("BedrockInlineAgent", 'Action Group & Knowledge Base', {
              actionGroups,
              knowledgeBases: kbs
            });

            this.logDebug("BedrockInlineAgent", 'Invoking Inline Agent', {
              foundationModel: this.foundationModel,
              enableTrace: this.enableTrace,
              sessionId
            });


            const command = new InvokeInlineAgentCommand({
              actionGroups,
              knowledgeBases: kbs,
              enableTrace: this.enableTrace,
              endSession: false,
              foundationModel: this.foundationModel,
              inputText: userRequest,
              instruction: description,
              sessionId: sessionId
            });

            let completion = "";
            const response = await this.bedrockAgentClient.send(command);

            // Process the response from the Amazon Bedrock agent
            if (response.completion === undefined) {
              throw new Error("Completion is undefined");
            }

            // Aggregate chunks of response data
            for await (const chunkEvent of response.completion) {
              if (chunkEvent.chunk) {
                const chunk = chunkEvent.chunk;
                const decodedResponse = new TextDecoder("utf-8").decode(chunk.bytes);
                completion += decodedResponse;
              } else if (this.enableTrace) {
                // Log chunk event details if tracing is enabled
                if (this.logger) this.logger.info("Chunk Event Details:", JSON.stringify(chunkEvent, null, 2));
              }
            }

            // Return the completed response as a Message object
            return {
              role: ParticipantRole.ASSISTANT,
              content: [{ text: completion }],
            };
          }
        }
      }

      throw new Error("Tool use block not handled");
    }

    async processRequest(
      inputText: string,
      userId: string,
      sessionId: string,
      chatHistory: ConversationMessage[],
      _additionalParams?: Record<string, string>
    ): Promise<ConversationMessage> {
      try {
        // Construct the user's message
        const userMessage: ConversationMessage = {
          role: ParticipantRole.USER,
          content: [{ text: inputText }]
        };

        // Combine chat history with current message
        const conversation: ConversationMessage[] = [...chatHistory, userMessage];

        this.updateSystemPrompt();


        this.logDebug("BedrockInlineAgent", 'System Prompt', this.systemPrompt);

        // Prepare the command to converse with the Bedrock API
        const converseCmd = {
          modelId: this.modelId,
          messages: conversation,
          system: [{ text: this.systemPrompt }],
          inferenceConfig: this.inferenceConfig,
          toolConfig: {
            tools: this.inlineAgentTool,
            toolChoice: {
              tool: {
                  name: BedrockInlineAgent.TOOL_NAME,
              },
          },
          },
        };


        this.logDebug("BedrockInlineAgent", 'Bedrock Command', JSON.stringify(converseCmd));


        // Call Bedrock's converse API
        const command = new ConverseCommand(converseCmd);
        const response = await this.client.send(command);

        if (!response.output) {
          throw new Error("No output received from Bedrock model");
        }

        const bedrockResponse = response.output.message as ConversationMessage;

        // Check if tool use is required
        if (bedrockResponse.content) {  // Add null check
          for (const content of bedrockResponse.content) {
            if (content && typeof content === 'object' && 'toolUse' in content) {
              return await this.toolConfig.useToolHandler(bedrockResponse, conversation, sessionId);
            }
          }
        }

        return bedrockResponse;

      } catch (error: unknown) {  // Explicitly type error as unknown
        // Handle error with proper type checking
        const errorMessage = error instanceof Error
          ? error.message
          : 'Unknown error occurred';

        if (this.logger) this.logger.error("Error processing request with Bedrock:", errorMessage);
        throw new Error(`Error processing request with Bedrock: ${errorMessage}`);
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
        ...this.customVariables
      };
      this.systemPrompt = this.replaceplaceholders(this.promptTemplate, allVariables);
    }

    private replaceplaceholders(template: string, variables: TemplateVariables): string {
      return template.replace(/{{(\w+)}}/g, (match, key) => {
        if (key in variables) {
          const value = variables[key];
          return Array.isArray(value) ? value.join('\n') : String(value);
        }
        return match;
      });
    }
  }