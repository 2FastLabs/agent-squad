# Shop MCP server (sample)

A tiny [MCP](https://modelcontextprotocol.io) server for the [ChatGPTStyleChat](../swift/ChatGPTStyleChat)
Swift sample. It exposes an order-lookup tool that advertises a **UI widget**, over streamable HTTP,
so the native app can point at it instead of its built-in in-app tool.

It shows the same contract ChatGPT Apps use: a tool returns `structuredContent` and advertises
`_meta.ui.resourceUri`. The app renders that widget as a native SwiftUI card, hydrated from the
tool's data — no HTML, no web view.

## Run it

```bash
# Optional: Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python shop_server.py                   # serves on http://127.0.0.1:8000/mcp (loopback)
HOST=0.0.0.0 python shop_server.py      # bind to the LAN, for a physical device
```

## Point the app at it

In the Swift sample, set the toggle in `Config.swift`:

```swift
static let mcpServerURL: String? = "http://127.0.0.1:8000/mcp"
```

Leave it `nil` to use the app's native `ShopToolProvider` instead. On the iOS Simulator,
`127.0.0.1` reaches your Mac. For a physical device, start the server with `HOST=0.0.0.0` (see
above) and use your Mac's LAN IP in the URL.

## Tools

| tool | visibility | notes |
|------|-----------|-------|
| `get_order(orderId)` | `model`, `app` | Order status + delivery estimate. Callable by the model; advertises the widget. |
| `refresh_order(orderId)` | `app` | Same data, marked app-only — never offered to the model. Only the widget's Refresh button calls it. |

Both return `structuredContent` (the render-only data the card hydrates from) and advertise
`_meta.ui.resourceUri = "ui://shop/order-card"`.

## Why the resource is a stub

The app renders the widget natively, so the served `ui://shop/order-card` resource is only a stub.
It must exist anyway, because the MCP client fetches the advertised resource when a tool advertises
one. A web host that wanted the widget as HTML would render that resource instead.
