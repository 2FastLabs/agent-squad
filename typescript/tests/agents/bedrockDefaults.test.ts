import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import {
  BedrockAgentRuntimeClient,
  InvokeInlineAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { BedrockLLMAgent } from "../../src/agents/bedrockLLMAgent";
import { BedrockInlineAgent } from "../../src/agents/bedrockInlineAgent";
import { BedrockTranslatorAgent } from "../../src/agents/bedrockTranslatorAgent";
import { BedrockClassifier } from "../../src/classifiers/bedrockClassifier";
import {
  BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET,
  BEDROCK_MODEL_ID_CLAUDE_3_HAIKU,
  BEDROCK_MODEL_ID_CLAUDE_3_SONNET,
  BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
} from "../../src/types";

const MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0";
const kinds = ["llm", "inline", "translator", "classifier"] as const;
type Kind = (typeof kinds)[number];

function makeAgent(kind: Kind, options: Record<string, any> = {}) {
  const common = {
    name: "Test",
    description: "Test requests",
    region: "us-east-1",
    ...options,
  };
  switch (kind) {
    case "inline":
      return new BedrockInlineAgent(common);
    case "translator":
      return new BedrockTranslatorAgent(common);
    case "classifier":
      return new BedrockClassifier(common);
    default:
      return new BedrockLLMAgent(common);
  }
}

async function sendRequest(agent: ReturnType<typeof makeAgent>) {
  if (agent instanceof BedrockClassifier) {
    agent.setAgents({});
    return agent.classify("hello", []);
  }
  return agent.processRequest("hello", "user", "session", []);
}

describe("Bedrock default model", () => {
  let send: jest.SpyInstance;

  beforeEach(() => {
    send = jest
      .spyOn(BedrockRuntimeClient.prototype, "send")
      .mockImplementation(async (command: any) => {
        if (command instanceof ConverseStreamCommand) {
          return {
            stream: (async function* () {
              yield { messageStop: { stopReason: "end_turn" } };
            })(),
          };
        }
        const tool = command.input.toolConfig?.tools?.[0]?.toolSpec?.name;
        let content: any[] = [{ text: "hello" }];
        if (tool === "Translate") {
          content = [
            { toolUse: { name: tool, input: { translation: "hello" } } },
          ];
        } else if (tool === "analyzePrompt") {
          content = [
            {
              toolUse: {
                name: tool,
                input: {
                  userinput: "hello",
                  selected_agent: "",
                  confidence: "0.8",
                },
              },
            },
          ];
        }
        return { output: { message: { role: "assistant", content } } };
      });
  });

  afterEach(() => jest.restoreAllMocks());

  it("exports the Claude Haiku 4.5 global profile and keeps the old constants", () => {
    expect(BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5).toBe(MODEL_ID);
    expect(BEDROCK_MODEL_ID_CLAUDE_3_HAIKU).toBe(
      "anthropic.claude-3-haiku-20240307-v1:0",
    );
    expect(BEDROCK_MODEL_ID_CLAUDE_3_SONNET).toBe(
      "anthropic.claude-3-sonnet-20240229-v1:0",
    );
    expect(BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET).toBe(
      "anthropic.claude-3-5-sonnet-20240620-v1:0",
    );
  });

  it.each(kinds)(
    "%s sends the Claude Haiku 4.5 profile by default",
    async (kind) => {
      await sendRequest(makeAgent(kind));

      const command = send.mock.calls[0][0] as ConverseCommand;
      expect(command).toBeInstanceOf(ConverseCommand);
      expect(command.input.modelId).toBe(MODEL_ID);
      const sampling = command.input.inferenceConfig ?? {};
      expect(
        sampling.temperature !== undefined && sampling.topP !== undefined,
      ).toBe(false);
    },
  );

  it.each(kinds)("%s sends an explicit modelId unchanged", async (kind) => {
    await sendRequest(
      makeAgent(kind, { modelId: BEDROCK_MODEL_ID_CLAUDE_3_HAIKU }),
    );

    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command.input.modelId).toBe(BEDROCK_MODEL_ID_CLAUDE_3_HAIKU);
  });

  it("BedrockLLMAgent streams with the Claude Haiku 4.5 profile by default", async () => {
    const stream = (await sendRequest(
      makeAgent("llm", { streaming: true }),
    )) as AsyncIterable<unknown>;
    for await (const _chunk of stream) {
      /* consume the stream */
    }

    const command = send.mock.calls[0][0] as ConverseStreamCommand;
    expect(command).toBeInstanceOf(ConverseStreamCommand);
    expect(command.input.modelId).toBe(MODEL_ID);
  });

  it("BedrockLLMAgent does not default temperature and topP together", async () => {
    await sendRequest(makeAgent("llm"));

    const { inferenceConfig } = (send.mock.calls[0][0] as ConverseCommand)
      .input;
    // Claude Haiku 4.5 rejects temperature and topP in the same request.
    expect(
      inferenceConfig.temperature !== undefined &&
        inferenceConfig.topP !== undefined,
    ).toBe(false);
  });

  it.each(kinds)("%s preserves explicit inference settings", async (kind) => {
    const inferenceConfig = { maxTokens: 500, topP: 0, stopSequences: ["END"] };
    await sendRequest(makeAgent(kind, { inferenceConfig }));

    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command.input.inferenceConfig).toEqual(
      expect.objectContaining(inferenceConfig),
    );
    expect(command.input.inferenceConfig.temperature).toBeUndefined();
    expect(inferenceConfig).toEqual({
      maxTokens: 500,
      topP: 0,
      stopSequences: ["END"],
    });
  });

  it("omits sampling parameters when thinking is enabled", async () => {
    const inferenceConfig = { maxTokens: 3000, temperature: 0.5, topP: 0.8 };
    const reasoningConfig = {
      thinking: { type: "enabled", budget_tokens: 2000 },
    };
    await sendRequest(makeAgent("llm", { inferenceConfig, reasoningConfig }));

    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command.input.inferenceConfig.maxTokens).toBe(3000);
    expect(command.input.inferenceConfig.temperature).toBeUndefined();
    expect(command.input.inferenceConfig.topP).toBeUndefined();
    expect(command.input.additionalModelRequestFields).toEqual(reasoningConfig);
    expect(inferenceConfig).toEqual({
      maxTokens: 3000,
      temperature: 0.5,
      topP: 0.8,
    });
  });

  it.each(["classifier", "translator"] as const)(
    "%s serializes inference settings into the HTTP body",
    async (kind) => {
      jest.restoreAllMocks();
      let body: any;
      const client = new BedrockRuntimeClient({
        region: "us-east-1",
        credentials: { accessKeyId: "test", secretAccessKey: "test" },
        requestHandler: {
          handle: async (request: any) => {
            body = JSON.parse(request.body);
            const tool = kind === "translator" ? "Translate" : "analyzePrompt";
            const input =
              kind === "translator"
                ? { translation: "hello" }
                : { userinput: "hello", selected_agent: "", confidence: 0.8 };
            return {
              response: {
                statusCode: 200,
                headers: { "content-type": "application/json" },
                body: new TextEncoder().encode(
                  JSON.stringify({
                    output: {
                      message: {
                        role: "assistant",
                        content: [
                          {
                            toolUse: { toolUseId: "tool-1", name: tool, input },
                          },
                        ],
                      },
                    },
                    stopReason: "tool_use",
                    usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
                    metrics: { latencyMs: 1 },
                  }),
                ),
              },
            };
          },
        },
      });
      const inferenceConfig = {
        maxTokens: 500,
        topP: 0.8,
        stopSequences: ["END"],
      };
      const agent = makeAgent(kind, { inferenceConfig });
      agent["client"] = client;
      try {
        await sendRequest(agent);
        expect(body.inferenceConfig).toEqual(inferenceConfig);
        expect(body).not.toHaveProperty("inferenceConfiguration");
      } finally {
        client.destroy();
      }
    },
  );

  describe("BedrockInlineAgent", () => {
    const inlineAgentToolUse = {
      output: {
        message: {
          role: "assistant",
          content: [
            {
              toolUse: {
                name: "inline_agent_creation",
                input: {
                  action_group_names: [],
                  knowledge_bases: [],
                  description: "Answer the question",
                  user_request: "hello",
                },
              },
            },
          ],
        },
      },
    };

    it("sends a temperature but no topP by default", async () => {
      await sendRequest(makeAgent("inline"));

      const command = send.mock.calls[0][0] as ConverseCommand;
      expect(command.input.inferenceConfig).toEqual({
        maxTokens: 1000,
        temperature: 0,
        stopSequences: [],
      });
    });

    it("sends the inferenceConfig it is given, topP included", async () => {
      const inferenceConfig = { maxTokens: 500, topP: 0.8 };

      await sendRequest(makeAgent("inline", { inferenceConfig }));

      const command = send.mock.calls[0][0] as ConverseCommand;
      expect(command.input.inferenceConfig).toEqual({
        maxTokens: 500,
        topP: 0.8,
      });
    });

    it.each([
      [undefined, MODEL_ID],
      ["custom-foundation-profile", "custom-foundation-profile"],
    ])(
      "passes the foundation model %s to the inline agent as %s",
      async (foundationModel, expected) => {
        send.mockResolvedValueOnce(inlineAgentToolUse);
        const inlineSend = jest
          .spyOn(BedrockAgentRuntimeClient.prototype, "send")
          .mockResolvedValue({
            completion: (async function* () {
              yield { chunk: { bytes: new TextEncoder().encode("hello") } };
            })(),
          } as never);

        await sendRequest(makeAgent("inline", { foundationModel }));

        const command = inlineSend.mock.calls[0][0] as InvokeInlineAgentCommand;
        expect(command).toBeInstanceOf(InvokeInlineAgentCommand);
        expect(command.input.foundationModel).toBe(expected);
      },
    );
  });
});
