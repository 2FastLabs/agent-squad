import {
  BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
  BedrockTranslatorAgent,
  BedrockTranslatorAgentOptions,
} from "../../src";
import { BedrockTranslatorAgent as Translator } from "../../src/agents/bedrockTranslatorAgent";

it("exports the migration model and translator through the package entry point", () => {
  const options: BedrockTranslatorAgentOptions = {
    name: "Translator",
    description: "Translate text",
    modelId: BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
  };

  expect(options.modelId).toBe(
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  );
  expect(BedrockTranslatorAgent).toBe(Translator);
});
