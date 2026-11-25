import readline from "readline";
import {
  AgentSquad,
  BedrockLLMAgent,
  AmazonBedrockAgent,
  LexBotAgent,
  LambdaAgent,
  Logger,
} from "agent-squad";

import {weatherToolDescription, weatherToolHanlder, WEATHER_PROMPT } from './tools/weather_tool'


function createOrchestrator(): AgentSquad {
  const orchestrator = new AgentSquad({
    config: {
      LOG_AGENT_CHAT: true,
      LOG_CLASSIFIER_CHAT: true,
      LOG_CLASSIFIER_RAW_OUTPUT: true,
      LOG_CLASSIFIER_OUTPUT: true,
      LOG_EXECUTION_TIMES: true,
      MAX_MESSAGE_PAIRS_PER_AGENT: 10,
    },
    logger: console,
  });

  // Add a Tech Agent to the orchestrator
  orchestrator.addAgent(
    new BedrockLLMAgent({
      name: "Tech Agent",
      description:
        "Specializes in technology areas including software development, hardware, AI, cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs related to technology products and services.",
      streaming: true,
      inferenceConfig: {
        temperature: 0.1,
      },
    })
  );

  // Add a Lex-based  Agent to the orchestrator
  orchestrator.addAgent(
    new LexBotAgent({
      name: "{{REPLACE_WITH_YOUR_AGENT_NAME}}",
      description: "{{REPLACE_WITH_YOUR_CUSTOM_AGENT_DESCRIPTION}}",
      botId: "{{LEX_BOT_ID}}",
      botAliasId: "{{LEX_BOT_ALIAS_ID}}",
      localeId: "{{LEX_BOT_LOCALE_ID}}",
    })
  );

  // Add weahter agent with tool
  const weatherAgent = new BedrockLLMAgent({
    name: "Weather Agent",
    modelId:"us.anthropic.claude-haiku-4-5-20251001-v1:0",
    description:
      "Specialized agent for giving weather condition from a city.",
    streaming: false,
    toolConfig: {
      tool: weatherToolDescription,
      useToolHandler: weatherToolHanlder,
      toolMaxRecursions: 5,
    },
    inferenceConfig: {
        temperature: 1.0,
        maxTokens:4096,
      },
      reasoningConfig:{
        thinking:{
          type:'enabled',
          budget_tokens: 4000,
        }
      }
  });
  weatherAgent.setSystemPrompt(WEATHER_PROMPT);
  orchestrator.addAgent(weatherAgent);

  // Add an Amazon Bedrock Agent to the orchestrator
  orchestrator.addAgent(
    new AmazonBedrockAgent({
      name: "{{AGENT_NAME}}",
      description: "{{REPLACE_WITH_YOUR_CUSTOM_AGENT_DESCRIPTION}}",
      agentId: "{{BEDROCK_AGENT_ID}}",
      agentAliasId: "{{BEDROCK_AGENT_ALIAS_ID}}",
    })
  );

  // Define your Lambda agents here
  const lambdaAgents = [
    {
      name: "{{LAMBDA_AGENT_NAME_1}}",
      description: "{{LAMBDA_AGENT_DESCRIPTION_1}}",
      functionName: "{{LAMBDA_FUNCTION_NAME_1}}",
      region: "{{AWS_REGION_1}}",
    },
    {
      name: "{{LAMBDA_AGENT_NAME_2}}",
      description: "{{LAMBDA_AGENT_DESCRIPTION_2}}",
      functionName: "{{LAMBDA_FUNCTION_NAME_2}}",
      region: "{{AWS_REGION_2}}",
    },
  ];
  // Add Lambda Agents to the orchestrator
  for (const agent of lambdaAgents) {
    orchestrator.addAgent(
      new LambdaAgent({
        name: agent.name,
        description: agent.description,
        functionName: agent.functionName,
        functionRegion: agent.region,
      })
    );
  }

  return orchestrator;
}

const uuidv4 = () => {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    var r = (Math.random() * 16) | 0,
      v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};
// Function to run local conversation
async function runLocalConversation(): Promise<void> {
  const orchestrator = createOrchestrator();
  // Generate random uuid 4

  const userId = uuidv4();
  const sessionId = uuidv4();

  const allAgents = orchestrator.getAllAgents();
  Logger.logger.log("Here are the existing agents:");
  for (const agentKey in allAgents) {
    const agent = allAgents[agentKey];
    Logger.logger.log(`Name: ${agent.name}`);
    Logger.logger.log(`Description: ${agent.description}`);
    Logger.logger.log("--------------------");
  }

  orchestrator.analyzeAgentOverlap();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  Logger.logger.log(
    "Welcome to the interactive AI agent. Type your queries and press Enter. Type 'exit' to end the conversation."
  );

  const askQuestion = (): void => {
    rl.question("You: ", async (userInput: string) => {
      if (userInput.toLowerCase() === "exit") {
        Logger.logger.log("Thank you for using the AI agent. Goodbye!");
        rl.close();
        return;
      }

      try {
        const response = await orchestrator.routeRequest(
          userInput,
          userId,
          sessionId
        );

        if (response.streaming == true) {
          Logger.logger.log("\n** RESPONSE STREAMING ** \n");
          // Send metadata immediately
          Logger.logger.log(`> Agent ID: ${response.metadata.agentId}`);
          Logger.logger.log(`> Agent Name: ${response.metadata.agentName}`);
          Logger.logger.log(`> User Input: ${response.metadata.userInput}`);
          Logger.logger.log(`> User ID: ${response.metadata.userId}`);
          Logger.logger.log(`> Session ID: ${response.metadata.sessionId}`);
          Logger.logger.log(
            `> Additional Parameters:`,
            response.metadata.additionalParams
          );
          Logger.logger.log(`\n> Response: `);

          // Stream the content
          for await (const chunk of response.output) {
            if (typeof chunk === "string") {
              process.stdout.write(chunk);
            }
            else if (typeof chunk === "object" && chunk.hasOwnProperty("thinking")) {
              // Print thinking content in cyan color
              process.stdout.write('\x1b[36m' + chunk.content + '\x1b[0m');
            } else {
              Logger.logger.error("Received unexpected chunk type:", typeof chunk);
            }
          }
          Logger.logger.log(); // Add a newline after the stream ends
          Logger.logger.log(); // Add a newline after the stream ends
        } else {
          // Handle non-streaming response (AgentProcessingResult)
          Logger.logger.log("\n** RESPONSE ** \n");
          Logger.logger.log(`> Agent ID: ${response.metadata.agentId}`);
          Logger.logger.log(`> Agent Name: ${response.metadata.agentName}`);
          Logger.logger.log(`> User Input: ${response.metadata.userInput}`);
          Logger.logger.log(`> User ID: ${response.metadata.userId}`);
          Logger.logger.log(`> Session ID: ${response.metadata.sessionId}`);
          Logger.logger.log(
            `> Additional Parameters:`,
            response.metadata.additionalParams
          );
          Logger.logger.log(`\n> Response: ${response.output}`);
        }
      } catch (error) {
        Logger.logger.error("Error:", error);
      }

      askQuestion(); // Continue the conversation
    });
  };

  askQuestion(); // Start the conversation
}

// Check if this script is being run directly (not imported as a module)
if (require.main === module) {
  // This block will only run when the script is executed locally
  runLocalConversation();
}
