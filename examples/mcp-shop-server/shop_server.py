"""A tiny MCP shop server for the ChatGPTStyleChat sample.

Exposes an order-lookup tool that advertises a UI widget, over streamable HTTP so a native app
(the Swift ChatGPTStyleChat sample) can point at it with `MCPServer(url: "http://…/mcp")`.

    pip install -r requirements.txt
    python shop_server.py            # serves on http://127.0.0.1:8000/mcp

Tools:
  get_order(orderId)     → order status + delivery estimate. Advertises the widget and is
                           callable by the model.
  refresh_order(orderId) → same, marked app-only (visibility ["app"]) so it is never offered to
                           the model — only the app's Refresh button calls it.

Both return `structuredContent` (the render-only data the native card hydrates from) and advertise
`_meta.ui.resourceUri = "ui://shop/order-card"`. The app renders that widget as native SwiftUI, so
the served `ui://` resource is only a stub — but it must exist, because the client fetches the
advertised resource when a tool advertises one.
"""

import os

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

ORDER_CARD_URI = "ui://shop/order-card"

# Loopback by default (Simulator reaches it). For a physical device, bind to the LAN with
# `HOST=0.0.0.0 python shop_server.py` and point the app at your Mac's LAN IP.
mcp = FastMCP("shop", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))


class Order(BaseModel):
    """The order shape. A typed return makes FastMCP emit it as `structuredContent` — the render-only
    data the native card hydrates from."""

    orderId: str
    status: str
    eta: str
    carrier: str
    items: list[str]


def _order(order_id: str, refreshed: bool) -> Order:
    """The faked backend — replace with a real API/database call."""
    return Order(
        orderId=order_id,
        status="Out for delivery" if refreshed else "In transit",
        eta="Today, 6:00 PM" if refreshed else "2026-07-02",
        carrier="FastShip",
        items=["Wireless Headphones", "USB-C Cable"],
    )


@mcp.tool(
    name="get_order",
    description="Look up an order's status and delivery estimate.",
    meta={"ui": {"resourceUri": ORDER_CARD_URI, "visibility": ["model", "app"]}},
    structured_output=True,
)
def get_order(orderId: str) -> Order:
    return _order(orderId, refreshed=False)


@mcp.tool(
    name="refresh_order",
    description="Refresh the order card with the latest status.",
    meta={"ui": {"resourceUri": ORDER_CARD_URI, "visibility": ["app"]}},  # app-only: hidden from the model
    structured_output=True,
)
def refresh_order(orderId: str) -> Order:
    return _order(orderId, refreshed=True)


@mcp.resource(ORDER_CARD_URI, mime_type="text/html;profile=mcp-app")
def order_card_resource() -> str:
    """The widget resource. The native app renders the card from `structuredContent` and ignores
    this body — it exists only because the client fetches the advertised resource. (A web host that
    wanted the widget as HTML would render this instead.)"""
    return "<!-- ui://shop/order-card — rendered natively by the app from structuredContent -->"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
