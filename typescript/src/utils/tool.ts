import { UIPayload } from "./ui";

// Internal wire wrapper (toolUseId + text) that formats a result for the model. Not the same as the
// public `ToolResult` below, which is what a tool returns.
export class AgentToolResult {
  constructor(
    public toolUseId: string,
    public content: any
  ) {}
}

/**
 * A tool return that carries render-only data and/or a UI widget. Only `content` (text) is added to
 * the model's context; `structuredContent` and `ui` never reach the model. Tools that need none of
 * this can still just return a string.
 */
export class ToolResult {
  constructor(
    public content: string = "",
    public structuredContent: any = {},
    public ui?: UIPayload
  ) {}
}

export class AgentToolCallbacks {
  /**
   * Defines callbacks that can be triggered during agent tool processing.
   */

  onToolStart(
      _toolName: string,
      _input: any,
      _runId?: string,
      _tags?: string[],
      _metadata?: Record<string, any>,
      ..._kwargs: any[]
  ): any {
      /**
       * Callback method that runs when a tool starts processing.
       *
       * @param toolName Name of the tool that is starting
       * @param input Object containing the tool's input
       * @param runId Unique identifier for this specific tool run
       * @param tags Optional list of string tags associated with this tool run
       * @param metadata Optional dictionary containing additional metadata about the run
       * @param kwargs Additional keyword arguments that might be passed to the callback
       * @returns The return value is implementation-dependent
       */
      // Default implementation does nothing
  }

  onToolEnd(
      _toolName: string,
      _input: any,
      _output: any,
      _runId?: string,
      _tags?: string[],
      _metadata?: Record<string, any>,
      ..._kwargs: any[]
  ): any {
      /**
       * Callback method that runs when a tool completes its processing.
       *
       * @param toolName Name of the tool that is completing
       * @param input Object containing the tool's input
       * @param output Object containing the tool's output
       * @param runId Unique identifier for this specific tool run
       * @param tags Optional list of string tags associated with this tool run
       * @param metadata Optional dictionary containing additional metadata about the run
       * @param kwargs Additional keyword arguments that might be passed to the callback
       * @returns The return value is implementation-dependent
       */
      // Default implementation does nothing
  }

  onToolError(
      _toolName: string,
      _input: any,
      _error: Error,
      _runId?: string,
      _tags?: string[],
      _metadata?: Record<string, any>,
      ..._kwargs: any[]
  ): any {
      /**
       * Callback method that runs when a tool encounters an error.
       *
       * @param toolName Name of the tool that encountered an error
       * @param input Object containing the tool's input
       * @param error The error that occurred
       * @param runId Unique identifier for this specific tool run
       * @param tags Optional list of string tags associated with this tool run
       * @param metadata Optional dictionary containing additional metadata about the run
       * @param kwargs Additional keyword arguments that might be passed to the callback
       * @returns The return value is implementation-dependent
       */
      // Default implementation does nothing
  }
}

type ToolFunction = (...args: any[]) => Promise<any> | any;

/**
 * Represents a single tool that can be used by an agent
 */
export class AgentTool {
  readonly name: string;
  readonly description: string;
  readonly properties: Record<string, any>;
  readonly required: string[];
  readonly func: ToolFunction;
  private enumValues: Record<string, any[]>;

  constructor(options: {
    name: string;
    description?: string;
    properties?: Record<string, any>;
    required?: string[];
    func: ToolFunction;
    enumValues?: Record<string, any[]>;
  }) {
    this.name = options.name;
    this.description =
      options.description || this.extractFunctionDescription();
    this.enumValues = options.enumValues || {};
    this.properties =
      options.properties || this.extractProperties(options.func);
    this.required = options.required || Object.keys(this.properties);
    this.func = this.wrapFunction(options.func);

    for (const [propName, enumVals] of Object.entries(this.enumValues)) {
      if (this.properties[propName]) {
        this.properties[propName].enum = enumVals;
      }
    }
  }

  private extractFunctionDescription(): string {
    return `Function to ${this.name}`;
  }

  private extractProperties(func: ToolFunction): Record<string, any> {
    const properties: Record<string, any> = {};
    const params = this.getParamNames(func);
    params.forEach((param) => {
      if (param === "this") return;
      properties[param] = {
        type: "string",
        description: `The ${param} parameter`,
      };
    });
    return properties;
  }

  private getParamNames(func: ToolFunction): string[] {
    const functionStr = func.toString();
    const paramStr = functionStr.slice(
      functionStr.indexOf("(") + 1,
      functionStr.indexOf(")")
    );
    return paramStr
      .split(",")
      .map((p) => p.trim())
      .filter((p) => p);
  }

  private wrapFunction(func: ToolFunction): ToolFunction {
    return async (...args: any[]) => {
      const result = func(...args);
      return result instanceof Promise ? await result : result;
    };
  }
}

/**
 * Manages a collection of tools that can be used by an agent
 */
export class AgentTools {
  public tools: AgentTool[];
  protected callbacks: AgentToolCallbacks;

  constructor(tools: AgentTool[], callbacks?: AgentToolCallbacks) {
    this.tools = tools;
    this.callbacks = callbacks ?? new AgentToolCallbacks()
  }

  async toolHandler(
    response: any,
    getToolUseBlock: (block: any) => any,
    getToolName: (toolUseBlock: any) => string,
    getToolId: (toolUseBlock: any) => string,
    getInputData: (toolUseBlock: any) => any
  ): Promise<AgentToolResult[]> {
    if (!response.content) {
      throw new Error("No content blocks in response");
    }

    const toolResults: AgentToolResult[] = [];
    const contentBlocks = response.content;

    for (const block of contentBlocks) {
      const toolUseBlock = getToolUseBlock(block);
      if (!toolUseBlock) continue;

      const toolName = getToolName(toolUseBlock);
      const toolId = getToolId(toolUseBlock);
      const inputData = getInputData(toolUseBlock);
      const result = await this.processTool(toolName, inputData);
      // A ToolResult carries render-only data/UI; only its text goes to the model. Empty text falls
      // back to the structured data so the model isn't blind. Plain returns pass through unchanged.
      const modelContent =
        result instanceof ToolResult
          ? result.content || JSON.stringify(result.structuredContent)
          : result;
      const toolResult = new AgentToolResult(toolId, modelContent);
      toolResults.push(toolResult);
    }

    return toolResults;
  }

  private async processTool(toolName: string, inputData: any): Promise<any> {
    try {
      let tool: AgentTool | undefined;
      for (const t of this.tools) {
        if (t.name === toolName) {
          tool = t;
          break;
        }
      }

      if (!tool?.func) {
        return `Tool '${toolName}' not found`;
      }

      if ("messages" in inputData) {
        return await tool.func(inputData.messages);
      }
      return await tool.func(inputData);
    } catch (error) {
      return `Error processing tool '${toolName}': ${error.message}`;
    }
  }
}