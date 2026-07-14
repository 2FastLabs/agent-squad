import { Agent, AgentOptions } from "../../src/agents/agent";
import {
  GroundedAgent,
  GroundedAgentOptions,
  DataBlockCurator,
  PerToolCurator,
  PresenterPrompt,
  Grounding,
  DEFAULT_PRESENTER_PROMPT,
  CapturedToolResult,
} from "../../src/agents/groundedAgent";
import { ConversationMessage, ParticipantRole } from "../../src/types";
import { AgentTool, AgentTools, ToolResult } from "../../src/utils/tool";
import { UIPolicy, UIPayload } from "../../src/utils/ui";

interface ToolCall {
  name: string;
  args: any;
}

class FakeGatherer extends Agent {
  lastHistory?: ConversationMessage[];
  constructor(
    options: AgentOptions,
    private readonly tools?: AgentTools,
    private readonly toolCalls: ToolCall[] = [],
    private readonly draft = "",
    private readonly streaming = false
  ) {
    super(options);
  }

  async processRequest(
    _inputText: string,
    _userId: string,
    _sessionId: string,
    chatHistory: ConversationMessage[]
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    this.lastHistory = chatHistory;
    for (const call of this.toolCalls) {
      const tool = this.tools!.tools.find((t) => t.name === call.name);
      await tool!.func(call.args);
    }
    if (this.streaming) {
      const draft = this.draft;
      return (async function* () {
        yield draft;
      })();
    }
    return { role: ParticipantRole.ASSISTANT, content: [{ text: this.draft }] };
  }
}

class FakePresenter extends Agent {
  systemPrompt?: string;
  receivedInput?: string;
  receivedHistory?: ConversationMessage[];
  called = false;
  constructor(
    options: AgentOptions,
    private readonly reply = "presented",
    private readonly streaming = false
  ) {
    super(options);
  }

  setSystemPrompt(template?: string): void {
    this.systemPrompt = template;
  }

  async processRequest(
    inputText: string,
    _userId: string,
    _sessionId: string,
    chatHistory: ConversationMessage[]
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    this.called = true;
    this.receivedInput = inputText;
    this.receivedHistory = chatHistory;
    if (this.streaming) {
      const reply = this.reply;
      return (async function* () {
        yield reply;
      })();
    }
    return { role: ParticipantRole.ASSISTANT, content: [{ text: this.reply }] };
  }
}

function makeTools(): AgentTools {
  return new AgentTools([new AgentTool({ name: "search_products", func: () => "unused" })]);
}

function build(
  gatherer: Agent,
  presenter: Agent,
  tools: AgentTools,
  extra: Partial<GroundedAgentOptions> = {}
): GroundedAgent {
  return new GroundedAgent({
    name: "Shop",
    description: "grounded shop",
    gatherer,
    presenter,
    tools,
    ...extra,
  });
}

async function toText(response: ConversationMessage | AsyncIterable<any>): Promise<string> {
  if (response && typeof (response as any)[Symbol.asyncIterator] === "function") {
    let text = "";
    for await (const chunk of response as AsyncIterable<any>) {
      text += typeof chunk === "string" ? chunk : chunk?.content?.[0]?.text ?? "";
    }
    return text;
  }
  const msg = response as ConversationMessage;
  return msg.content?.[0]?.text ?? "";
}

describe("GroundedAgent", () => {
  // --- Tool turn -> presenter runs, grounded on the curated feed ---

  it("runs the presenter with the curated feed on a tool turn", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: { query: "x" } }],
      "brain draft that must not surface"
    );
    // Real result comes from the tool's func; override it for this test.
    (tools.tools[0] as any).func = () => "Blue shoes, €90";
    const presenter = new FakePresenter({ name: "p", description: "" }, "Blue shoes cost €90.");
    const agent = build(gatherer, presenter, tools);

    const result = await agent.processRequest("shoes?", "u", "s", []);

    expect(await toText(result)).toBe("Blue shoes cost €90.");
    expect(presenter.receivedInput).toContain("<user question>\nshoes?");
    expect(presenter.receivedInput).toContain("### search_products\nBlue shoes, €90");
    expect(presenter.receivedInput).not.toContain("brain draft");
  });

  it("resolves the presenter prompt from the primary tool", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "get_order", args: {} }]
    );
    (tools.tools[0] as any).name = "get_order"; // ensure the captured name matches
    const presenter = new FakePresenter({ name: "p", description: "" });
    const agent = build(gatherer, presenter, tools, {
      presenterPrompt: new PresenterPrompt("DEFAULT", { get_order: "ORDER PROMPT" }),
    });

    await agent.processRequest("where is my order?", "u", "s", []);
    expect(presenter.systemPrompt).toBe("ORDER PROMPT");
  });

  it("falls back to the default presenter prompt for unmapped tools", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }]
    );
    const presenter = new FakePresenter({ name: "p", description: "" });
    const agent = build(gatherer, presenter, tools, {
      presenterPrompt: new PresenterPrompt("DEFAULT", { other: "X" }),
    });

    await agent.processRequest("q", "u", "s", []);
    expect(presenter.systemPrompt).toBe("DEFAULT");
  });

  it("forwards chat history to the presenter", async () => {
    const tools = makeTools();
    const history: ConversationMessage[] = [
      { role: ParticipantRole.USER, content: [{ text: "earlier" }] },
    ];
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }]
    );
    const presenter = new FakePresenter({ name: "p", description: "" });
    const agent = build(gatherer, presenter, tools);

    await agent.processRequest("q", "u", "s", history);
    expect(presenter.receivedHistory).toBe(history);
  });

  // --- No-tool turn -> skip presenter, emit the gatherer's own reply ---

  it("skips the presenter on a no-tool turn", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer({ name: "g", description: "" }, tools, [], "Hi there!");
    const presenter = new FakePresenter({ name: "p", description: "" });
    const agent = build(gatherer, presenter, tools);

    const result = await agent.processRequest("hello", "u", "s", []);
    expect(await toText(result)).toBe("Hi there!");
    expect(presenter.called).toBe(false);
  });

  // --- Capture seam restores the original tool functions ---

  it("restores the original tool functions after the turn", async () => {
    const tools = makeTools();
    const original = tools.tools[0].func;
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }]
    );
    const presenter = new FakePresenter({ name: "p", description: "" });
    const agent = build(gatherer, presenter, tools);

    await agent.processRequest("q", "u", "s", []);
    expect(tools.tools[0].func).toBe(original);
  });

  // --- Streaming path ---

  it("passes the presenter's stream straight through on a tool turn", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }],
      "",
      true
    );
    const presenter = new FakePresenter({ name: "p", description: "" }, "streamed", true);
    const agent = build(gatherer, presenter, tools);

    const result = await agent.processRequest("q", "u", "s", []);
    expect(typeof (result as any)[Symbol.asyncIterator]).toBe("function");
    expect(await toText(result)).toBe("streamed");
  });

  it("returns the gatherer draft on a streaming no-tool turn", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [],
      "just chatting",
      true
    );
    const presenter = new FakePresenter({ name: "p", description: "" }, "unused", true);
    const agent = build(gatherer, presenter, tools);

    const result = await agent.processRequest("hi", "u", "s", []);
    expect(await toText(result)).toBe("just chatting");
    expect(presenter.called).toBe(false);
  });

  // --- Tool UI (widget) forwarding ---

  const uiToolResult = () => {
    const payload: UIPayload = {
      resourceUri: "ui://shop/order-card",
      mimeType: "text/html;profile=mcp-app",
      template: "<div>card</div>",
      structuredContent: { status: "shipped" },
    };
    return new ToolResult("Order: shipped", { status: "shipped" }, payload);
  };

  it("forwards the primary tool's widget before the presenter text on a tool turn", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }],
      "",
      true
    );
    (tools.tools[0] as any).func = () => uiToolResult();
    const presenter = new FakePresenter({ name: "p", description: "" }, "Your order shipped.", true);
    const agent = build(gatherer, presenter, tools);

    const result = await agent.processRequest("order?", "u", "s", []);
    const chunks: any[] = [];
    for await (const chunk of result as AsyncIterable<any>) chunks.push(chunk);

    expect(chunks[0].ui.resourceUri).toBe("ui://shop/order-card");
    expect(chunks[chunks.length - 1]).toBe("Your order shipped.");
    // The presenter is fed the structured facts, not the widget wrapper.
    expect(presenter.receivedInput).toContain('"status": "shipped"');
  });

  it("emits no widget when uiPolicy is SUPPRESS", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }],
      "",
      true
    );
    (tools.tools[0] as any).func = () => uiToolResult();
    const presenter = new FakePresenter({ name: "p", description: "" }, "text only", true);
    const agent = build(gatherer, presenter, tools, { uiPolicy: UIPolicy.SUPPRESS });

    const result = await agent.processRequest("q", "u", "s", []);
    const chunks: any[] = [];
    for await (const chunk of result as AsyncIterable<any>) chunks.push(chunk);
    expect(chunks.some((c) => c && typeof c === "object" && c.ui)).toBe(false);
    expect(chunks.join("")).toBe("text only");
  });

  it("curates a ToolResult from its structuredContent", () => {
    const out = new DataBlockCurator().curate([
      { name: "get_product", result: new ToolResult("ignored text", { price: 90 }) },
    ]);
    expect(out).toContain("### get_product");
    expect(out).toContain('"price": 90');
    expect(out).not.toContain("ignored text");
  });

  it("curates from content when a ToolResult has empty structuredContent", () => {
    // `{}` is truthy in JS — verifies the explicit-emptiness check, not a naive `sc || content`.
    const out = new DataBlockCurator().curate([{ name: "t", result: new ToolResult("plain text") }]);
    expect(out).toBe("### t\nplain text");
  });

  it("returns a plain message with no widget for a non-streaming presenter", async () => {
    const tools = makeTools();
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [{ name: "search_products", args: {} }]
    );
    (tools.tools[0] as any).func = () => uiToolResult();
    const presenter = new FakePresenter({ name: "p", description: "" }, "Your order shipped.");
    const agent = build(gatherer, presenter, tools);

    const result = await agent.processRequest("order?", "u", "s", []);
    // Non-streaming: a ConversationMessage (not a stream), and no widget channel.
    expect(typeof (result as any)[Symbol.asyncIterator]).toBe("undefined");
    expect((result as ConversationMessage).content?.[0].text).toBe("Your order shipped.");
  });

  it("Grounding.primaryUi returns the primary tool's widget", () => {
    const tr = uiToolResult();
    expect(Grounding.primaryUi([{ name: "get_order", result: tr }])?.resourceUri).toBe(
      "ui://shop/order-card"
    );
    // A trailing non-UI tool must not hide an earlier widget (prefers the last UI call; Swift parity).
    expect(
      Grounding.primaryUi([
        { name: "get_order", result: tr },
        { name: "check_stock", result: "in stock" },
      ])?.resourceUri
    ).toBe("ui://shop/order-card");
    // No UI tool at all → no widget.
    expect(
      Grounding.primaryUi([
        { name: "a", result: "plain" },
        { name: "b", result: "plain" },
      ])
    ).toBeUndefined();
    expect(Grounding.primaryUi([])).toBeUndefined();
  });

  it("forwards the widget and keys the prompt off the UI tool when a non-UI tool trails it", async () => {
    // A UI tool followed by a plain non-UI helper.
    const tools = new AgentTools([
      new AgentTool({ name: "search_products", func: () => uiToolResult() }),
      new AgentTool({ name: "check_stock", func: () => "in stock" }),
    ]);
    const gatherer = new FakeGatherer(
      { name: "g", description: "" },
      tools,
      [
        { name: "search_products", args: {} },
        { name: "check_stock", args: {} },
      ],
      "",
      true
    );
    const presenter = new FakePresenter({ name: "p", description: "" }, "done", true);
    const agent = build(gatherer, presenter, tools, {
      presenterPrompt: new PresenterPrompt("DEFAULT", { search_products: "PRODUCT PROMPT" }),
    });

    const result = await agent.processRequest("q", "u", "s", []);
    const chunks: any[] = [];
    for await (const chunk of result as AsyncIterable<any>) chunks.push(chunk);

    expect(chunks.filter((c) => c && typeof c === "object" && c.ui)).toHaveLength(1);
    expect(presenter.systemPrompt).toBe("PRODUCT PROMPT");
  });

  // --- Construction guards ---

  it("requires a gatherer and a presenter", () => {
    const tools = makeTools();
    expect(
      () =>
        new GroundedAgent({
          name: "x",
          description: "",
          gatherer: undefined as any,
          presenter: new FakePresenter({ name: "p", description: "" }),
          tools,
        })
    ).toThrow();
  });

  it("requires tools", () => {
    expect(
      () =>
        new GroundedAgent({
          name: "x",
          description: "",
          gatherer: new FakeGatherer({ name: "g", description: "" }),
          presenter: new FakePresenter({ name: "p", description: "" }),
          tools: undefined as any,
        })
    ).toThrow();
  });
});

describe("Curators", () => {
  it("DataBlockCurator formats string and structured results", () => {
    const out = new DataBlockCurator().curate([
      { name: "search", result: "raw text" },
      { name: "detail", result: { price: 90, name: "shoe" } },
    ]);
    expect(out).toContain("### search\nraw text");
    expect(out).toContain("### detail\n");
    expect(out).toContain('"name": "shoe"');
    expect(out).toContain("\n\n");
  });

  it("PerToolCurator routes and falls back", () => {
    const curator = new PerToolCurator({
      search: (tool: CapturedToolResult) => `CUSTOM:${tool.result}`,
    });
    const out = curator.curate([
      { name: "search", result: "hits" },
      { name: "other", result: "plain" },
    ]);
    expect(out).toContain("CUSTOM:hits");
    expect(out).toContain("### other\nplain");
  });
});

describe("PresenterPrompt / Grounding", () => {
  it("PresenterPrompt.default resolves the default text", () => {
    expect(PresenterPrompt.default().resolve()).toBe(DEFAULT_PRESENTER_PROMPT);
    expect(PresenterPrompt.default().resolve("anything")).toBe(DEFAULT_PRESENTER_PROMPT);
  });

  it("Grounding.primary is the last call", () => {
    const results: CapturedToolResult[] = [
      { name: "a", result: 1 },
      { name: "b", result: 2 },
    ];
    expect(Grounding.primary(results)?.name).toBe("b");
    expect(Grounding.primary([])).toBeUndefined();
  });

  it("Grounding.presenterMessage tags the question and data", () => {
    expect(Grounding.presenterMessage("Q", "D")).toBe(
      "<user question>\nQ\n</user question>\n<data to present>\nD\n</data to present>"
    );
  });
});
