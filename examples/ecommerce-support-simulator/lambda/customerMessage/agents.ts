import {
  AgentSquad,
  BedrockLLMAgent,
  BedrockLLMAgentOptions,
  AmazonBedrockAgent,
  AmazonBedrockAgentOptions,
  AgentResponse,
  AmazonKnowledgeBasesRetriever,
  Agent,
  ChainAgent,
  ChainAgentOptions,
  ConversationMessage,
  ParticipantRole,
  AnthropicClassifier,
} from "agent-squad";
import { BedrockAgentRuntimeClient } from "@aws-sdk/client-bedrock-agent-runtime";
import { SQSClient, SendMessageCommand } from "@aws-sdk/client-sqs";
import * as fs from 'fs';

// Load configuration from JSON file
const config = JSON.parse(fs.readFileSync('mock_data.json', 'utf-8'));

// Mock databases
const orderDb: Record<string, any> = config.orders;
const shipmentDb: Record<string, any> = config.shipments;

// Mock functions for tools
const orderLookup = (orderId: string): any => {
  console.log(`OrderLookup - Order ID: ${orderId}`);
  const result = orderDb[orderId] || "not found";
  console.log(`OrderLookup - Result: ${JSON.stringify(result)}`);
  return result;
};

const shipmentTracker = (orderId: string): any => {
  console.log(`ShipmentTracker - Order ID: ${orderId}`);
  const result = shipmentDb[orderId] || "not found";
  console.log(`ShipmentTracker - Result: ${JSON.stringify(result)}`);
  return result;
};

const returnProcessor = (orderId: string): string => {
  console.log(`ReturnProcessor - Order ID: ${orderId}`);
  const result = `Return initiated for order ${orderId}`;
  console.log(`ReturnProcessor - Result: ${result}`);
  return result;
};

const orderManagementToolConfig = [
  {
    toolSpec: {
      name: "orderlookup",
      description: "Retrieve order details from the database",
      inputSchema: {
        json: {
          type: "object",
          properties: {
            orderId: { type: "string", description: "The order ID to look up" },
          },
          required: ["orderId"],
        },
      },
    },
  },
  {
    toolSpec: {
      name: "shipmenttracker",
      description: "Get real-time shipping information",
      inputSchema: {
        json: {
          type: "object",
          properties: {
            orderId: { type: "string", description: "The order ID to track" },
          },
          required: ["orderId"],
        },
      },
    },
  },
  {
    toolSpec: {
      name: "returnprocessor",
      description: "Initiate and manage return requests",
      inputSchema: {
        json: {
          type: "object",
          properties: {
            orderId: {
              type: "string",
              description: "The order ID for the return",
            },
          },
          required: ["orderId"],
        },
      },
    },
  },
];

async function orderManagementToolHandler(
  response: ConversationMessage,
  conversation: ConversationMessage[]
) {
  console.log('Starting orderManagementToolHandler');
  console.log('Response:', JSON.stringify(response, null, 2));
  console.log('Conversation:', JSON.stringify(conversation, null, 2));

  const responseContentBlocks = response.content as any[];
  let toolResults: any = [];

  console.log('Response content blocks:', JSON.stringify(responseContentBlocks, null, 2));

  if (!responseContentBlocks) {
    console.error('No content blocks in response');
    throw new Error("No content blocks in response");
  }

  for (const contentBlock of responseContentBlocks) {
    console.log('Processing content block:', JSON.stringify(contentBlock, null, 2));

    if ("toolUse" in contentBlock) {
      const toolUseBlock = contentBlock.toolUse;
      const toolUseName = toolUseBlock.name;
      console.log('Tool use detected:', toolUseName);

      let result;
      switch (toolUseName) {
        case "orderlookup":
          console.log('Executing OrderLookup with orderId:', toolUseBlock.input.orderId);
          result = orderLookup(toolUseBlock.input.orderId);
          break;
        case "shipmenttracker":
          console.log('Executing ShipmentTracker with orderId:', toolUseBlock.input.orderId);
          result = shipmentTracker(toolUseBlock.input.orderId);
          break;
        case "returnprocessor":
          console.log('Executing ReturnProcessor with orderId:', toolUseBlock.input.orderId);
          result = returnProcessor(toolUseBlock.input.orderId);
          break;
      }

      console.log('Tool execution result:', JSON.stringify(result, null, 2));

      if (result) {
        toolResults.push({
          toolResult: {
            toolUseId: toolUseBlock.toolUseId,
            content: [{ json: { result } }],
          },
        });
      }
    }
  }

  console.log('Final tool results:', JSON.stringify(toolResults, null, 2));

  const message: ConversationMessage = {
    role: ParticipantRole.USER,
    content: toolResults,
  };

  console.log('New message to be added to conversation:', JSON.stringify(message, null, 2));

  console.log('Updated conversation:', JSON.stringify(conversation, null, 2));
  console.log('orderManagementToolHandler completed');

  return message;
}

export const orderManagementAgent = new BedrockLLMAgent({
  name: "Order Management Agent",
  streaming: false,
  description:
    "Handles order-related inquiries including order status, shipment tracking, returns, and refunds. Uses order database and shipment tracking tools.",
  modelId: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  toolConfig: {
    useToolHandler: orderManagementToolHandler,
    tool: orderManagementToolConfig,
    toolMaxRecursions: 5,
  },
  saveChat: false,
} as BedrockLLMAgentOptions);

const productInfoRetriever = new AmazonKnowledgeBasesRetriever(
  new BedrockAgentRuntimeClient({}),
  { knowledgeBaseId: "your-product-kb-id" }
);

export const customerServiceAgent = new AmazonBedrockAgent({
  name: "Customer Service Agent",
  streaming: false,
  description:
    "Handles general customer inquiries, account-related questions, and non-technical support requests. Uses comprehensive customer service knowledge base.",
  agentId: "your-agent-id",
  agentAliasId: "your-agent-alias-id",
  saveChat: false,
} as AmazonBedrockAgentOptions);

export const productInfoAgent = new BedrockLLMAgent({
  name: "Product Information Agent",
  streaming: false,
  description:
    "Provides detailed product information, answers questions about specifications, compatibility, and availability.",
  modelId: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  retriever: productInfoRetriever,
  saveChat: false,
} as BedrockLLMAgentOptions);

export interface HumanAgentOptions {
  name: string;
  description: string;
  saveChat: boolean;
  queueUrl: string;
}

export class HumanAgent extends Agent {
  private sqsClient: SQSClient;
  private queueUrl: string;

  constructor(options: HumanAgentOptions) {
    super(options);
    console.log("Human agent init");
    this.sqsClient = new SQSClient({});
    this.queueUrl = options.queueUrl;
    if (!this.queueUrl) {
      throw new Error("Queue URL is not provided");
    }

  }

  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[]
  ): Promise<ConversationMessage> {
    console.log(`Human agent received request: ${inputText}`);
    const humanResponse = `Your request "${inputText}" has been received and will be processed by our customer service team. We'll get back to you as soon as possible.`;

    // Send messages to SQS
    //await this.sendSQSMessage("customer", humanResponse);
    await this.sendSQSMessage("support", inputText);

    return {
      role: ParticipantRole.ASSISTANT,
      content: [{ text: humanResponse }],
    };
  }

  private async sendSQSMessage(destination: string, message: string) {
    const command = new SendMessageCommand({
      QueueUrl: this.queueUrl,
      MessageBody: JSON.stringify({
        destination,
        message: message,
      }),
    });

    try {
      const response = await this.sqsClient.send(command);
      console.log(`Message sent to ${destination}:`, response.MessageId);
    } catch (error) {
      console.error(`Error sending message to ${destination}:`, error);
    }
  }
}

export const aiWithHumanVerificationAgent = new ChainAgent({
  name: "AI with Human Verification Agent",
  description:
    "Handles high-priority or sensitive customer inquiries by generating AI responses and having them verified by a human before sending.",
  agents: [
    customerServiceAgent,
    new HumanAgent({
      name: "Human Verifier",
      description: "Verifies and potentially modifies AI-generated responses",
      saveChat: false,
      queueUrl: process.env.QUEUE_URL || "",
    }),
  ],
  saveChat: false,
} as ChainAgentOptions);


