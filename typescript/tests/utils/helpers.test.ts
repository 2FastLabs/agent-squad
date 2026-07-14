import { AccumulatorTransform } from "../../src/utils/helpers";

describe("AccumulatorTransform", () => {
  it("forwards a { ui } widget chunk without folding it into the accumulated text", async () => {
    const transform = new AccumulatorTransform();
    const out: any[] = [];
    transform.on("data", (c) => out.push(c));
    const done = new Promise<void>((resolve) => transform.on("end", () => resolve()));

    transform.write("Hello ");
    transform.write({ ui: { resourceUri: "ui://x", mimeType: "text/html;profile=mcp-app" } });
    transform.write("world");
    transform.end();
    await done;

    // The saved text answer excludes the widget object.
    expect(transform.getAccumulatedData()).toBe("Hello world");
    // The widget object is forwarded to the stream consumer...
    const widget = out.find((c) => c && typeof c === "object" && c.ui);
    expect(widget.ui.resourceUri).toBe("ui://x");
    // ...alongside the text chunks.
    expect(out.filter((c) => typeof c === "string").join("")).toBe("Hello world");
  });

  it("accumulates and forwards plain text chunks unchanged", async () => {
    const transform = new AccumulatorTransform();
    const out: string[] = [];
    transform.on("data", (c) => out.push(c));
    const done = new Promise<void>((resolve) => transform.on("end", () => resolve()));

    transform.write("a");
    transform.write("b");
    transform.end();
    await done;

    expect(transform.getAccumulatedData()).toBe("ab");
    expect(out.join("")).toBe("ab");
  });

  it("does not treat a chunk with a falsy .ui as a widget", async () => {
    const transform = new AccumulatorTransform();
    const out: any[] = [];
    transform.on("data", (c) => out.push(c));
    const done = new Promise<void>((resolve) => transform.on("end", () => resolve()));

    transform.write({ ui: undefined }); // falsy ui → text path → dropped like any unknown chunk
    transform.write("text");
    transform.end();
    await done;

    expect(transform.getAccumulatedData()).toBe("text");
    expect(out.some((c) => c && typeof c === "object")).toBe(false); // no object forwarded
  });
});
