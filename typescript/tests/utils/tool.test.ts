import { AgentTools, AgentTool, AgentToolResult, ToolResult } from "../../src/utils/tool";

const getToolUseBlock = (block: any) => block.toolUse ?? null;
const getToolName = (b: any) => b.name;
const getToolId = (b: any) => b.toolUseId;
const getInputData = (b: any) => b.input;

function bedrockResponse(toolName: string, toolUseId: string, input: any) {
  return { role: "assistant", content: [{ toolUse: { name: toolName, toolUseId, input } }] };
}

describe("AgentTools.toolHandler ToolResult routing", () => {
  it("routes only a ToolResult's text to the model", async () => {
    const tools = new AgentTools([
      new AgentTool({
        name: "get_order",
        func: (input: any) => new ToolResult(`Order ${input.orderId}`, { id: input.orderId }),
      }),
    ]);

    const results = await tools.toolHandler(
      bedrockResponse("get_order", "1", { orderId: "42" }),
      getToolUseBlock,
      getToolName,
      getToolId,
      getInputData
    );

    expect(results[0]).toBeInstanceOf(AgentToolResult);
    expect(results[0].content).toBe("Order 42"); // structured data + widget never reach the model
  });

  it("falls back to structuredContent JSON when a ToolResult has no text", async () => {
    const tools = new AgentTools([
      new AgentTool({ name: "t", func: (input: any) => new ToolResult("", { x: input.x }) }),
    ]);

    const results = await tools.toolHandler(
      bedrockResponse("t", "1", { x: "hi" }),
      getToolUseBlock,
      getToolName,
      getToolId,
      getInputData
    );

    expect(results[0].content).toBe(JSON.stringify({ x: "hi" }));
  });

  it("passes a plain string return through unchanged", async () => {
    const tools = new AgentTools([new AgentTool({ name: "t", func: () => "just text" })]);

    const results = await tools.toolHandler(
      bedrockResponse("t", "1", {}),
      getToolUseBlock,
      getToolName,
      getToolId,
      getInputData
    );

    expect(results[0].content).toBe("just text");
  });
});
