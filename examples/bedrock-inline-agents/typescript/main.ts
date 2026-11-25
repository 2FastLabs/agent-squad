//import { BedrockInlineAgent, BedrockInlineAgentOptions } from 'agent-squad';
import { BedrockInlineAgent, BedrockInlineAgentOptions } from '../../../typescript/src/agents/bedrockInlineAgent';
import {
  BedrockAgentRuntimeClient,
  AgentActionGroup,
  KnowledgeBase
} from "@aws-sdk/client-bedrock-agent-runtime";
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";
import { v4 as uuidv4 } from 'uuid';
import { createInterface } from 'readline';

// Define action groups
const actionGroupsList: AgentActionGroup[] = [
  {
    actionGroupName: 'CodeInterpreterAction',
    parentActionGroupSignature: 'AMAZON.CodeInterpreter',
    description: 'Use this to write and execute python code to answer questions and other tasks.'
  },
  {
    actionGroupExecutor: {
      lambda: "arn:aws:lambda:region:0123456789012:function:my-function-name"
    },
    actionGroupName: "MyActionGroupName",
    apiSchema: {
      s3: {
        s3BucketName: "bucket-name",
        s3ObjectKey: "openapi-schema.json"
      }
    },
    description: "My action group for doing a specific task"
  }
];

// Define knowledge bases
const knowledgeBases: KnowledgeBase[] = [
  {
    knowledgeBaseId: "knowledge-base-id-01",
    description: 'This is my knowledge base for documents 01',
  },
  {
    knowledgeBaseId: "knowledge-base-id-02",
    description: 'This is my knowledge base for documents 02',
  },
  {
    knowledgeBaseId: "knowledge-base-id-03",
    description: 'This is my knowledge base for documents 03',
  }
];


// Initialize BedrockInlineAgent
const bedrickInlineAgent = new BedrockInlineAgent({
  name: "Inline Agent Creator for Agents for Amazon Bedrock",
  region: 'us-east-1',
  modelId: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  description: "Specialized in creating Agent to solve customer request dynamically. You are provided with a list of Action groups and Knowledge bases which can help you in answering customer request",
  actionGroupsList: actionGroupsList,
  knowledgeBases: knowledgeBases,
  LOG_AGENT_DEBUG_TRACE: true
});

async function runInlineAgent(userInput: string, userId: string, sessionId: string) {
  const response = await bedrickInlineAgent.processRequest(
    userInput,
    userId,
    sessionId,
    [], // empty chat history
    undefined // no additional params
  );
  return response;
}

async function main() {
  const sessionId = uuidv4();
  const userId = uuidv4();

  console.log("Welcome to the interactive Multi-Agent system. Type 'quit' to exit.");

  const readline = createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const getUserInput = () => {
    return new Promise((resolve) => {
      readline.question('\nYou: ', (input) => {
        resolve(input.trim());
      });
    });
  };

  while (true) {
    const userInput = await getUserInput() as string;

    if (userInput.toLowerCase() === 'quit') {
      console.log("Exiting the program. Goodbye!");
      readline.close();
      process.exit(0);
    }

    try {
      const response = await runInlineAgent(userInput, userId, sessionId);
      if (response && response.content && response.content.length > 0) {
        const text = response.content[0]?.text;
        console.log(text || 'No response content');
      } else {
        console.log('No response');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  }
}

// Run the program
main().catch(console.error);