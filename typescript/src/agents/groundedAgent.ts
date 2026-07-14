import { Agent, AgentOptions } from "./agent";
import { ConversationMessage, ParticipantRole } from "../types";
import { AgentTool, AgentTools, ToolResult } from "../utils/tool";
import { UIPayload, UIPolicy } from "../utils/ui";

/**
 * The 2-LLM grounded (anti-hallucination) pattern as an {@link Agent}.
 *
 * A *gatherer* LLM calls tools and sees the raw results but never speaks to the user; an isolated
 * *presenter* LLM writes the reply grounded only on the curated facts — no tools, no tool responses —
 * so it cannot invent values beyond what was fetched. A chit-chat turn that calls no tools is answered
 * in one pass by the gatherer, skipping the presenter.
 *
 * When a tool returns a {@link ToolResult} carrying a UI widget, the primary tool's widget is
 * forwarded to the caller as a `{ ui }` chunk on the streaming response, governed by `uiPolicy`.
 */

/** The generic grounding instruction: present only the provided data, never invent values. */
export const DEFAULT_PRESENTER_PROMPT =
  "You are presenting information to the user. Use ONLY the data provided. Be concise and " +
  "natural, and never invent or infer values that are not present in the data.";

/** One captured tool call, as the curator sees it. */
export interface CapturedToolResult {
  name: string;
  result: any;
}

/**
 * Turns gathered tool results into the text the presenter is fed — GroundedAgent's data extension
 * point; default is {@link DataBlockCurator}. A pure synchronous transform by contract, no I/O: a
 * curator needing external data pre-fetches it and is constructed with it.
 */
export interface ToolOutputCurator {
  curate(results: CapturedToolResult[]): string;
}

/** Recursively re-serializes with sorted object keys, so structured results are stable/diffable. */
function stableStringify(value: any): string {
  return JSON.stringify(
    value,
    (_key, val) =>
      val && typeof val === "object" && !Array.isArray(val)
        ? Object.keys(val)
            .sort()
            .reduce((acc: Record<string, any>, k) => {
              acc[k] = val[k];
              return acc;
            }, {})
        : val,
    2
  );
}

/**
 * The default curator: a faithful `### <toolName>` section per tool — the raw string result, or the
 * structured result pretty-printed as JSON — concatenated across every captured tool.
 */
export class DataBlockCurator implements ToolOutputCurator {
  curate(results: CapturedToolResult[]): string {
    return results.map((result) => DataBlockCurator.section(result)).join("\n\n");
  }

  /** One section. Also used as {@link PerToolCurator}'s fallback formatter. */
  static section(result: CapturedToolResult): string {
    let value = result.result;
    if (value instanceof ToolResult) {
      // Curate from the render-only structured data (the facts), not the widget wrapper. `{}` is
      // truthy in JS, so check emptiness explicitly rather than `structuredContent || content`.
      const sc = value.structuredContent;
      value = sc && typeof sc === "object" && Object.keys(sc).length ? sc : value.content;
    }
    const body = typeof value === "string" ? value : stableStringify(value);
    return `### ${result.name}\n${body}`;
  }
}

/** Formats one captured tool into its section of the feed. */
export type ToolFormatter = (tool: CapturedToolResult) => string;

/**
 * A curator routing each captured tool to its own formatter, keyed by tool name (the key
 * {@link PresenterPrompt} uses), with a fallback for unmapped tools. Each formatter trims/compacts
 * its tool's section, so this is where you shrink an oversized payload before the presenter sees it.
 */
export class PerToolCurator implements ToolOutputCurator {
  constructor(
    private readonly formatters: Record<string, ToolFormatter>,
    private readonly fallback: ToolFormatter = DataBlockCurator.section
  ) {}

  curate(results: CapturedToolResult[]): string {
    return results
      .map((result) => (this.formatters[result.name] ?? this.fallback)(result))
      .join("\n\n");
  }
}

/**
 * Chooses the presenter's system prompt, keyed by the turn's primary tool. One prompt by default;
 * supply a per-tool map for tool-specific presenters.
 */
export class PresenterPrompt {
  constructor(
    private readonly defaultPrompt: string,
    private readonly perTool: Record<string, string> = {}
  ) {}

  /** The prompt for a turn whose primary tool is `primaryTool` (falls back to the default). */
  resolve(primaryTool?: string): string {
    if (primaryTool && this.perTool[primaryTool]) {
      return this.perTool[primaryTool];
    }
    return this.defaultPrompt;
  }

  static default(): PresenterPrompt {
    return new PresenterPrompt(DEFAULT_PRESENTER_PROMPT);
  }
}

/** Shared helpers for the gather -> present pattern. */
export const Grounding = {
  /**
   * The turn's primary tool: the last call that advertised a UI widget, else the last call (matches
   * Swift). Drives both the presenter prompt and the forwarded widget, so a widget still surfaces
   * when a non-UI helper runs after the UI tool.
   */
  primary(results: CapturedToolResult[]): CapturedToolResult | undefined {
    for (let i = results.length - 1; i >= 0; i--) {
      const r = results[i];
      if (r.result instanceof ToolResult && r.result.ui) return r;
    }
    return results.length ? results[results.length - 1] : undefined;
  },

  /** The widget advertised by the primary tool, if it returned one. */
  primaryUi(results: CapturedToolResult[]): UIPayload | undefined {
    const primary = Grounding.primary(results);
    return primary && primary.result instanceof ToolResult ? primary.result.ui : undefined;
  },

  /**
   * The presenter's user message: question and curated data, tagged so the model can tell them
   * apart.
   */
  presenterMessage(question: string, data: string): string {
    return (
      "<user question>\n" +
      question +
      "\n</user question>\n" +
      "<data to present>\n" +
      data +
      "\n</data to present>"
    );
  },
};

export interface GroundedAgentOptions extends AgentOptions {
  /** The brain agent — configured with `tools` and its own system prompt; runs the tool loop and never speaks to the user. */
  gatherer: Agent;
  /** The presenter agent — should have no tools; writes the grounded reply. May be a cheaper/smaller model. */
  presenter: Agent;
  /** The AgentTools the gatherer uses. GroundedAgent wraps them to capture results. */
  tools: AgentTools;
  /** Shapes gathered results into the presenter feed. Default: DataBlockCurator. */
  curator?: ToolOutputCurator;
  /** Per-tool presenter system prompts. Default: a generic grounding prompt. */
  presenterPrompt?: PresenterPrompt;
  /** Whether a tool-advertised widget is forwarded to the caller (streaming path only). Default: forward. */
  uiPolicy?: UIPolicy;
}

/**
 * The 2-LLM grounded pattern: a gatherer calls tools, an isolated presenter answers only from the
 * curated facts. A no-tool turn is answered by the gatherer directly, skipping the presenter.
 */
export class GroundedAgent extends Agent {
  private readonly gatherer: Agent;
  private readonly presenter: Agent;
  private readonly tools: AgentTools;
  private readonly curator: ToolOutputCurator;
  private readonly presenterPrompt: PresenterPrompt;
  private readonly uiPolicy: UIPolicy;

  constructor(options: GroundedAgentOptions) {
    super(options);
    if (!options.gatherer || !options.presenter) {
      throw new Error("GroundedAgent requires both a gatherer and a presenter agent.");
    }
    if (!options.tools) {
      throw new Error("GroundedAgent requires the AgentTools the gatherer uses.");
    }
    this.gatherer = options.gatherer;
    this.presenter = options.presenter;
    this.tools = options.tools;
    this.curator = options.curator ?? new DataBlockCurator();
    this.presenterPrompt = options.presenterPrompt ?? PresenterPrompt.default();
    this.uiPolicy = options.uiPolicy ?? UIPolicy.FORWARD;
  }

  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    const { draft, captured } = await this.gather(
      inputText,
      userId,
      sessionId,
      chatHistory,
      additionalParams
    );

    // No tools called -> chit-chat: speak the gatherer's own reply, skip the presenter.
    if (captured.length === 0) {
      return { role: ParticipantRole.ASSISTANT, content: [{ text: draft }] };
    }

    // Curate the facts, select the presenter prompt from the primary tool, run the presenter —
    // which has no tools and speaks only from history + question + feed. Its response (message or
    // stream) is returned as-is, so presenter streaming flows straight through.
    const feed = this.curator.curate(captured);
    const primary = Grounding.primary(captured);
    const prompt = this.presenterPrompt.resolve(primary?.name);
    const presenter = this.presenter as Agent & {
      setSystemPrompt?: (template?: string) => void;
    };
    if (typeof presenter.setSystemPrompt === "function") {
      presenter.setSystemPrompt(prompt);
    } else {
      this.logger.warn(
        `Presenter ${this.presenter.name} has no setSystemPrompt; PresenterPrompt ignored.`
      );
    }
    const presenterInput = Grounding.presenterMessage(inputText, feed);
    const presenterResponse = await this.presenter.processRequest(
      presenterInput,
      userId,
      sessionId,
      chatHistory,
      additionalParams
    );

    // Forward the primary tool's widget (if any) as a leading `{ ui }` chunk, then the presenter's
    // text. Streaming path only — a non-streaming presenter returns a message with no widget channel.
    const widget = this.uiPolicy === UIPolicy.FORWARD ? Grounding.primaryUi(captured) : undefined;
    if (widget && GroundedAgent.isAsyncIterable(presenterResponse)) {
      return (async function* () {
        yield { ui: widget };
        yield* presenterResponse as AsyncIterable<any>;
      })();
    }
    return presenterResponse;
  }

  /**
   * Run the gatherer's tool loop with each tool's function wrapped to capture its result, then
   * restore the originals. Its own draft reply is returned (used only if no tools were called).
   */
  private async gather(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<{ draft: string; captured: CapturedToolResult[] }> {
    const captured: CapturedToolResult[] = [];
    const toolList = this.tools.tools;
    const originals = toolList.map((tool) => tool.func);
    toolList.forEach((tool: AgentTool) => {
      const original = tool.func;
      // `func` is readonly at compile time; the wrap is per-turn state, restored in `finally`.
      (tool as any).func = async (...args: any[]) => {
        const result = await original(...args);
        captured.push({ name: tool.name, result });
        return result;
      };
    });

    try {
      const response = await this.gatherer.processRequest(
        inputText,
        userId,
        sessionId,
        chatHistory,
        additionalParams
      );
      const draft = await GroundedAgent.drain(response);
      return { draft, captured };
    } finally {
      toolList.forEach((tool, index) => {
        (tool as any).func = originals[index];
      });
    }
  }

  /** Collect the final text from either a complete message or a stream of string/message chunks. */
  private static async drain(
    response: ConversationMessage | AsyncIterable<any>
  ): Promise<string> {
    if (!GroundedAgent.isAsyncIterable(response)) {
      const content = (response as ConversationMessage).content;
      return content && content.length ? content[0].text ?? "" : "";
    }
    let text = "";
    for await (const chunk of response as AsyncIterable<any>) {
      if (typeof chunk === "string") {
        text += chunk;
      } else if (chunk?.content?.[0]?.text) {
        text = chunk.content[0].text;
      }
    }
    return text;
  }

  private static isAsyncIterable(obj: any): obj is AsyncIterable<any> {
    return obj && typeof obj[Symbol.asyncIterator] === "function";
  }
}
