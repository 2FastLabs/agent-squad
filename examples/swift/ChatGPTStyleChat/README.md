# Chatbots, Meet Your UI — sample app

The complete, runnable companion to the article **"Chatbots, Meet Your UI: Building ChatGPT-Style
Chat Apps on iOS."**

A native SwiftUI chat that answers in **text *and* interactive widgets**, built on the
[Agent Squad](https://github.com/2FastLabs/agent-squad) Swift framework. The assistant uses a
**GroundedAgent** (a *Brain* that fetches via tools + a *Presenter* that speaks only from those
facts), and a tool that returns a `UIPayload` the app renders as a **native SwiftUI view**, chosen
by `resourceURI` and hydrated from the tool's data.

What this sample gives you on top of the article's snippets:

- The full **wiring** (`@main` App → `ChatViewModel` → `Orchestrator` → `GroundedAgent` → `ShopToolProvider`).
- A **Widgets on/off toggle** in the header — the same agent in `ui: .forward` vs `ui: .suppress`,
  so you can see *text + widget* vs *text only* side by side.
- The complete **Refresh round-trip**: the card's button calls an `.app`-only tool (invisible to the
  LLM) directly in Swift; the fresh `UIPayload` replaces the old one and the SwiftUI view re-renders.

## Requirements

- Xcode 26+ with the **Swift 6.2** toolchain
- iOS 17+ simulator or device (the sample uses `DeviceChatStorage`, which is SwiftData; for iOS 16
  swap it for `InMemoryChatStorage()` — see `ChatViewModel.swift`)
- An OpenAI API key (or any OpenAI-compatible endpoint)

## Run it (2 minutes)

This project is ready to open — no setup, no `Add Package` step. It references the Agent Squad
package by **local path** (`../../..`, the repo root), so it always builds against the checked-out
framework.

1. `open examples/swift/ChatGPTStyleChat/ChatGPTStyleChat.xcodeproj`
2. **Set your key**: either add `OPENAI_API_KEY` to the scheme's *Run → Environment Variables*, or
   paste it into `Config.swift`. **Don't commit a real key.**
3. Pick an **iOS 17+ simulator** and hit **Run**.
4. Try: `where is my order #1234?`

> Using this outside the repo? Delete the local package reference and add the remote one instead —
> *File → Add Package Dependencies…* → `https://github.com/2FastLabs/agent-squad`, branch `main`,
> product **AgentSquad**.

## Try this

- Ask **"where is my order #1234?"** → an order card appears, with a grounded sentence underneath.
- Tap **Refresh** on the card → the status flips to *Out for delivery* with a new ETA. Notice the LLM
  was never involved — the widget called the `.app`-only `refresh_order` tool directly.
- Flip the **Widgets** toggle off and ask again → the *same* grounded answer, as text only.

## Tools from a remote MCP server (optional)

By default the tools are native, in-app Swift code (`ShopToolProvider`). The same chat works
unchanged against a remote **MCP server** — the GroundedAgent doesn't care where tools live. Set the
toggle in `Config.swift`:

```swift
static let mcpServerURL: String? = "http://127.0.0.1:8000/mcp"   // nil = native ShopToolProvider
```

A runnable server is included at [`examples/mcp-shop-server`](../../mcp-shop-server) — it exposes the
same `get_order` / `refresh_order` tools and advertises the same `ui://shop/order-card` widget, so
the native `OrderCardView` renders identically. Start it with `python shop_server.py`, set the URL
above, and run. On the Simulator `127.0.0.1` reaches your Mac; for a physical device start the server
with `HOST=0.0.0.0 python shop_server.py` and use your Mac's LAN IP in the URL.

## How it maps to the article

| File | Article section |
|------|-----------------|
| `ChatGPTStyleChatApp.swift` | the `@main` entry point |
| `Config.swift` | the API key plumbing (Prerequisites) + the optional `mcpServerURL` toggle |
| `ShopToolProvider.swift` | **Step 3** — a tool that returns a widget (`get_order`) + an `.app`-only tool (`refresh_order`) |
| `OrderCardView.swift` | **Step 6** — the native SwiftUI widget for `ui://shop/order-card`, hydrated from `structuredContent` |
| `ChatViewModel.swift` | **Step 2 & 4** — the GroundedAgent, prompts, and the `.forward`/`.suppress` toggle; the widget→app callback |
| `ChatView.swift` | **Step 5** — the SwiftUI chat |
| `WidgetView.swift` | **Step 6** — maps a `UIPayload`'s `resourceURI` to a native SwiftUI view (add one `case` per widget) |

## Notes

- The "backend" in `ShopToolProvider` is faked in-memory so the sample runs with zero infrastructure.
  Replace `order(_:refreshed:)` with your real API/database call.
- **Tool visibility is enforced by the provider.** `ShopToolProvider.listTools()` filters to
  model-visible tools (`visibility.contains(.model)`), so the `.app`-only `refresh_order` is never
  advertised to the LLM — it stays callable directly via `call(_:arguments:)`. The built-in `ToolKit`
  does this filtering for you; a custom provider must do it itself.
- `structuredContent` hydrates the widget and is not added to the model's context (the model reasons
  from the tool's `content` text), so the card can't be hallucinated from.
- The whole thing runs **on device**. Point `ChatCompletionsClient` at a local model and you get
  offline, "free-token" inference (see `ChatViewModel.swift`).
