import { SupervisorAgent } from "../../src/agents/supervisorAgent";
import { BedrockLLMAgent } from "../../src/agents/bedrockLLMAgent";
import { ConversationMessage, ParticipantRole } from "../../src/types";

class MockBedrockLLMAgent extends BedrockLLMAgent {
  constructor(config: { name: string; description: string }) {
    super(config);
  }

  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    return {
      role: ParticipantRole.ASSISTANT,
      content: [{ text: "Mock response" }],
    };
  }
}

describe("SupervisorAgent", () => {
  describe("constructor", () => {
    it("should preserve custom name and description instead of overwriting them with the lead agent's values", () => {
      const leadAgent = new MockBedrockLLMAgent({
        name: "Lead Agent Name",
        description: "Lead Agent Description",
      });

      const supervisor = new SupervisorAgent({
        name: "Custom Supervisor Name",
        description: "Custom Supervisor Description",
        leadAgent,
        team: [],
      });

      expect(supervisor.name).toBe("Custom Supervisor Name");
      expect(supervisor.description).toBe("Custom Supervisor Description");
    });

    it("should fall back to the lead agent's name and description when none are provided", () => {
      const leadAgent = new MockBedrockLLMAgent({
        name: "Lead Agent Name",
        description: "Lead Agent Description",
      });

      const supervisor = new SupervisorAgent({
        name: "",
        description: "",
        leadAgent,
        team: [],
      });

      expect(supervisor.name).toBe("Lead Agent Name");
      expect(supervisor.description).toBe("Lead Agent Description");
    });
  });
});
