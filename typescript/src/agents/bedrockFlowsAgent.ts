import { BedrockAgentRuntimeClient, InvokeFlowCommand } from "@aws-sdk/client-bedrock-agent-runtime";
import { Agent, AgentOptions } from "./agent";
import {
  ConversationMessage,
  ParticipantRole
} from "../types";
import { addUserAgentMiddleware } from '../common/src/awsSdkUtils';

  export interface BedrockFlowsAgentOptions extends AgentOptions {
    region?: string;
    flowIdentifier: string;
    flowAliasIdentifier: string;
    bedrockAgentClient?: BedrockAgentRuntimeClient;
    enableTrace?: boolean;
    flowInputEncoder?: (agent: Agent, inputText: string, ...kwargs: any) => any;
    flowOutputDecoder?: (agent: Agent, response: any) => any;
  }

  export class BedrockFlowsAgent extends Agent {

    /** Protected class members */
    protected flowIdentifier: string;
    protected flowAliasIdentifier: string;
    protected bedrockAgentClient: BedrockAgentRuntimeClient;
    protected enableTrace: boolean;
    private flowInputEncoder: (agent: Agent, inputText: string, params: any) => any; // Accepts any additional properties
    private flowOutputDecoder: (agent: Agent, response: any) => ConversationMessage;

    constructor(options: BedrockFlowsAgentOptions) {
      super(options);

      this.bedrockAgentClient = options.bedrockAgentClient ?? (
        options.region
          ? new BedrockAgentRuntimeClient({ region: options.region })
          : new BedrockAgentRuntimeClient()
      );

      addUserAgentMiddleware(this.bedrockAgentClient, "bedrock-flows-agent");

      this.flowIdentifier = options.flowIdentifier;
      this.flowAliasIdentifier = options.flowAliasIdentifier;
      this.enableTrace = options.enableTrace ?? false;
      if (options.flowInputEncoder) {
        this.flowInputEncoder = options.flowInputEncoder;
      } else {
        this.flowInputEncoder = this.defaultFlowInputEncoder;
      }

      if (options.flowOutputDecoder){
        this.flowOutputDecoder = options.flowOutputDecoder;
      } else {
        this.flowOutputDecoder = this.defaultFlowOutputDecoder;
      }
    }

    defaultFlowInputEncoder(agent: Agent, inputText: string, ..._kwargs: any): any {
      return  inputText
    }

    defaultFlowOutputDecoder(agent: Agent, response: any): ConversationMessage {
      return {
        role: ParticipantRole.ASSISTANT,
        content: [{ text: response }],
      };
    }

    async processRequest(
      inputText: string,
      userId: string,
      sessionId: string,
      chatHistory: ConversationMessage[],
      additionalParams?: Record<string, string>
    ): Promise<ConversationMessage> {

      try {
        // Prepare the command to invoke Bedrock Flows
        const invokeFlowCommand = new InvokeFlowCommand({
          flowIdentifier:this.flowIdentifier,
          flowAliasIdentifier:this.flowAliasIdentifier,
          inputs: [
            {
                'content': {
                    'document': this.flowInputEncoder(this, inputText, {userId:userId, sessionId:sessionId, chatHistory:chatHistory, ...additionalParams})
                },
                'nodeName': 'FlowInputNode',
                'nodeOutputName': 'document'
            }
          ],
          ...(this.enableTrace && { enableTrace: true }),
        });

        let completion;
        // Call Bedrock's Invoke Flow API
        const response = await this.bedrockAgentClient.send(invokeFlowCommand);

        if (!response.responseStream) {
          throw new Error("No output received from Bedrock Llows");
        }

        // Aggregate chunks of response data
        for await (const flowEvent of response.responseStream) {
          if (flowEvent.flowOutputEvent) {
            completion = flowEvent.flowOutputEvent.content.document;
          } else if (this.enableTrace) {
            // Log chunk event details if tracing is enabled
            if (this.logger) this.logger.info("Flow Event Details:", JSON.stringify(flowEvent, null, 2));
          }
        }

        // Return the completed response as a Message object
        return this.flowOutputDecoder(this, completion);

      } catch (error: unknown) {  // Explicitly type error as unknown
      // Handle error with proper type checking
      const errorMessage = error instanceof Error
        ? error.message
        : 'Unknown error occurred';

      if (this.logger) this.logger.error("Error processing request with Bedrock:", errorMessage);
      throw new Error(`Error processing request with Bedrock: ${errorMessage}`);
    }
  }
}