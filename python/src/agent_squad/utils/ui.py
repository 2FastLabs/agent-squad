"""Tool-UI (widget) types.

A tool can return a self-contained UI component (an "MCP App" widget) alongside its text. The host
renders it from ``structured_content``, which is render-only: never added to the model's context, so
the model cannot hallucinate from it. These types carry the data only — drawing the widget is the
host's job.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class UISecurity:
    """Security the host enforces when rendering a component (built into a CSP). Undeclared domains
    are blocked."""

    connect_domains: list[str] = field(default_factory=list)
    resource_domains: list[str] = field(default_factory=list)
    frame_domains: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)  # e.g. "camera", "microphone"
    domain: Optional[str] = None                          # dedicated sandbox origin
    prefers_border: bool = False


@dataclass
class UIPayload:
    """A widget package advertised by a tool. ``structured_content``/``meta`` are render-only —
    never added to the model's context.

    ``template`` is the component body (HTML, a URL, or a remote-DOM script); the kind is given by
    ``mime_type``.
    """

    resource_uri: str                                     # e.g. "ui://shop/order-card"
    mime_type: str                                        # e.g. "text/html;profile=mcp-app"
    template: Optional[str] = None
    structured_content: Any = field(default_factory=dict)
    meta: Optional[dict[str, Any]] = None
    security: Optional[UISecurity] = None


class UIPolicy(str, Enum):
    """Whether an advertised tool UI is surfaced to the caller (decided per agent)."""

    FORWARD = "forward"    # surface the widget on the response (default)
    SUPPRESS = "suppress"  # fold the data into the text answer; emit no widget
