import { BedrockAgentRuntimeClient, InvokeAgentCommand, InvokeAgentCommandOutput } from "@aws-sdk/client-bedrock-agent-runtime";
import { ConversationMessage, ParticipantRole } from "../types";
import { Agent, AgentOptions } from "./agent";
import { Logger } from "../utils/logger";
import { addUserAgentMiddleware } from '../common/src/awsSdkUtils';

/**
 * Options for configuring an Amazon Bedrock agent.
 * Extends base AgentOptions with specific parameters required for Amazon Bedrock.
 */
export interface AmazonBedrockAgentOptions extends AgentOptions {
  region?: string;
  agentId: string;        // The ID of the Amazon Bedrock agent.
  agentAliasId: string;   // The alias ID of the Amazon Bedrock agent.
  client?: BedrockAgentRuntimeClient;  // Client for interacting with the Bedrock agent runtime.
  enableTrace?: boolean;  // Flag to enable tracing of Agent
  streaming?: boolean;    // Flag to enable streaming of responses.
}


/**
 * Represents an Amazon Bedrock agent that interacts with a runtime client.
 * Extends base Agent class and implements specific methods for Amazon Bedrock.
 */
export class AmazonBedrockAgent extends Agent {
  private agentId: string;                    // The ID of the Amazon Bedrock agent.
  private agentAliasId: string;               // The alias ID of the Amazon Bedrock agent.
  private client: BedrockAgentRuntimeClient;  // Client for interacting with the Bedrock agent runtime.
  private enableTrace: boolean;// Flag to enable tracing of Agent
  private streaming: boolean;    // Flag to enable streaming of responses.

  /**
   * Constructs an instance of AmazonBedrockAgent with the specified options.
   * Initializes the agent ID, agent alias ID, and creates a new Bedrock agent runtime client.
   * @param options - Options to configure the Amazon Bedrock agent.
   */
  constructor(options: AmazonBedrockAgentOptions) {
    super(options);
    this.agentId = options.agentId;
    this.agentAliasId = options.agentAliasId;
    this.client = options.client ? options.client : options.region
    ? new BedrockAgentRuntimeClient({ region: options.region })
    : new BedrockAgentRuntimeClient();
    addUserAgentMiddleware(this.client, "bedrock-agent");
    this.enableTrace = options.enableTrace || false;
    this.streaming = options.streaming || false;
  }

  private async *handleStreamingResponse(response: InvokeAgentCommandOutput): AsyncIterable<string> {
    for await (const chunkEvent of response.completion) {
      if (chunkEvent.chunk) {
        const chunk = chunkEvent.chunk;
        const decodedResponse = new TextDecoder("utf-8").decode(chunk.bytes);
        yield decodedResponse;
      } else if (chunkEvent.trace){
        if (this.enableTrace){
          Logger.logger.info("Trace:", JSON.stringify(chunkEvent.trace));
        }
      }
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
    additionalParams?: Record<any, any>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    // Construct the command to invoke the Amazon Bedrock agent with user input
    const command = new InvokeAgentCommand({
      agentId: this.agentId,
      agentAliasId: this.agentAliasId,
      sessionId: sessionId,
      inputText: inputText,
      sessionState: additionalParams ? additionalParams.sessionState?  additionalParams.sessionState : undefined : undefined,
      enableTrace: this.enableTrace,
      streamingConfigurations: {
        streamFinalResponse: this.streaming,
      }
    });

    try {
      let completion = "";
      const response = await this.client.send(command);

      // Process the response from the Amazon Bedrock agent
      if (response.completion === undefined) {
        throw new Error("Completion is undefined");
      }

      if (this.streaming){
        return this.handleStreamingResponse(response);
      } else {
        const citations: any[] = [];
        // Aggregate chunks of response data
        for await (const chunkEvent of response.completion) {
          if (chunkEvent.chunk) {
            const chunk = chunkEvent.chunk;
            const decodedResponse = new TextDecoder("utf-8").decode(chunk.bytes);
            completion += decodedResponse;
            if (chunk.attribution?.citations) {
              citations.push(...chunk.attribution.citations);
            }
          } else if (chunkEvent.trace) {
            if (this.enableTrace){
              Logger.logger.info("Trace:", JSON.stringify(chunkEvent.trace));
            }
          }
        }

        // Return the completed response as a Message object
        return {
          role: ParticipantRole.ASSISTANT,
          content: [{ text: completion }],
          ...(citations.length > 0 && { citations }),
        };
      }
    } catch (err) {
      // Handle errors encountered while invoking the Amazon Bedrock agent
      Logger.logger.error(err);
      throw err;
    }
  }
}

