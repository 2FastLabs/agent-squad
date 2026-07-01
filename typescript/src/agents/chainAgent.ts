import { Agent, AgentOptions } from "./agent";
import { ConversationMessage, ParticipantRole } from "../types";
import { Logger } from "../utils/logger";

export interface ChainAgentOptions extends AgentOptions {
  agents: Agent[];
  defaultOutput?: string;
}

export class ChainAgent extends Agent {
  agents: Agent[];
  private defaultOutput: string;

  constructor(options: ChainAgentOptions) {
    super(options);
    this.agents = options.agents;
    this.defaultOutput = options.defaultOutput || "No output generated from the chain.";

    if (this.agents.length === 0) {
      throw new Error("ChainAgent requires at least one agent in the chain.");
    }
  }

/**
   * Processes a user request by sending it to the Amazon Bedrock agent for processing.
   * @param inputText - The user input as a string.
   * @param userId - The ID of the user sending the request.
   * @param sessionId - The ID of the session associated with the conversation.
   * @param chatHistory - An array of Message objects representing the conversation history.
   * @param additionalParams - Optional additional parameters as key-value pairs.
   * @returns A Promise that resolves to a Message object containing the agent's response.
   */
  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {

    let currentInput = inputText;
    let finalResponse: ConversationMessage | AsyncIterable<any>;

    console.log(`Processing chain with ${this.agents.length} agents`);

    for (let i = 0; i < this.agents.length; i++) {
      const isLastAgent = i === this.agents.length - 1;
      const agent = this.agents[i];

      try {
        console.log(`Input for agent ${i}: ${currentInput}`);
        const response = await agent.processRequest(
          currentInput,
          userId,
          sessionId,
          chatHistory,
          additionalParams
        );

        if (this.isConversationMessage(response)) {
          if (response.content.length > 0 && 'text' in response.content[0]) {
            currentInput = response.content[0].text;
            finalResponse = response;
            console.log(`Output from agent ${i}: ${currentInput}`);
          } else {
            Logger.logger.warn(`Agent ${agent.name} returned no text content.`);
            return this.createDefaultResponse();
          }
        } else if (this.isAsyncIterable(response)) {
          if (!isLastAgent) {
            Logger.logger.warn(`Intermediate agent ${agent.name} returned a streaming response, which is not allowed.`);
            return this.createDefaultResponse();
          }
          // It's the last agent and streaming is allowed
          finalResponse = response;
        } else {
          Logger.logger.warn(`Agent ${agent.name} returned an invalid response type.`);
          return this.createDefaultResponse();
        }

        // If it's not the last agent, ensure we have a non-streaming response to pass to the next agent
        if (!isLastAgent && !this.isConversationMessage(finalResponse)) {
          Logger.logger.error(`Expected non-streaming response from intermediate agent ${agent.name}`);
          return this.createDefaultResponse();
        }
      } catch (error) {
        Logger.logger.error(`Error processing request with agent ${agent.name}:`, error);
        throw new Error(`Error processing request with agent ${agent.name}:${String(error)}`);
      }
    }

    return finalResponse;
  }

  private isAsyncIterable(obj: any): obj is AsyncIterable<any> {
    return obj && typeof obj[Symbol.asyncIterator] === 'function';
  }


  private isConversationMessage(response: any): response is ConversationMessage {
    return response && 'role' in response && 'content' in response && Array.isArray(response.content);
  }

  private createDefaultResponse(): ConversationMessage {
    return {
      role: ParticipantRole.ASSISTANT,
      content: [{ text: this.defaultOutput }],
    };
  }
}