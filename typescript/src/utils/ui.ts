/**
 * Tool-UI (widget) types.
 *
 * A tool can return a self-contained UI component (an "MCP App" widget) alongside its text. The host
 * renders it from `structuredContent`, which is render-only: never added to the model's context, so
 * the model cannot hallucinate from it. These types carry the data only — drawing the widget is the
 * host's job.
 */

/** Security the host enforces when rendering a component (built into a CSP). Undeclared domains are blocked. */
export interface UISecurity {
  connectDomains?: string[];
  resourceDomains?: string[];
  frameDomains?: string[];
  permissions?: string[]; // e.g. "camera", "microphone"
  domain?: string; // dedicated sandbox origin
  prefersBorder?: boolean;
}

/**
 * A widget package advertised by a tool. `structuredContent`/`meta` are render-only — never added to
 * the model's context. `template` is the component body (HTML, a URL, or a remote-DOM script); the
 * kind is given by `mimeType`.
 */
export interface UIPayload {
  resourceUri: string; // e.g. "ui://shop/order-card"
  mimeType: string; // e.g. "text/html;profile=mcp-app"
  template?: string;
  structuredContent?: any;
  meta?: Record<string, any>;
  security?: UISecurity;
}

/** Whether an advertised tool UI is surfaced to the caller (decided per agent). */
export enum UIPolicy {
  FORWARD = "forward", // surface the widget on the response (default)
  SUPPRESS = "suppress", // fold the data into the text answer; emit no widget
}
