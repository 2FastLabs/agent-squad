"""Unit tests for GroundedAgent (the 2-LLM grounded pattern)."""
import pytest
from unittest.mock import AsyncMock

from agent_squad.agents import (
    Agent,
    AgentOptions,
    AgentStreamResponse,
    GroundedAgent,
    GroundedAgentOptions,
    DataBlockCurator,
    PerToolCurator,
    PresenterPrompt,
    CapturedToolResult,
)
from agent_squad.agents.grounded_agent import Grounding, DEFAULT_PRESENTER_PROMPT
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils import AgentTool, AgentTools


def _msg(text: str) -> ConversationMessage:
    return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": text}])


class _FakeGatherer(Agent):
    """A gatherer whose ``process_request`` optionally drives the tools' callbacks (simulating a
    real agent's tool loop) then returns a draft message. Records the args it was called with."""

    def __init__(self, options, tools=None, tool_calls=None, draft="", streaming=False):
        super().__init__(options)
        self._tools = tools
        self._tool_calls = tool_calls or []  # list of (tool_name, input, output)
        self._draft = draft
        self._streaming = streaming
        self.last_history = None

    def is_streaming_enabled(self):
        return self._streaming

    async def process_request(self, input_text, user_id, session_id, chat_history, additional_params=None):
        self.last_history = chat_history
        for name, payload, output in self._tool_calls:
            await self._tools.callbacks.on_tool_start(name, payload)
            await self._tools.callbacks.on_tool_end(name, payload, output)
        if self._streaming:
            async def gen():
                yield AgentStreamResponse(text=self._draft, final_message=_msg(self._draft))
            return gen()
        return _msg(self._draft)


class _FakePresenter(Agent):
    def __init__(self, options, reply="presented", streaming=False):
        super().__init__(options)
        self._reply = reply
        self._streaming = streaming
        self.system_prompt = None
        self.received_input = None
        self.received_history = None

    def is_streaming_enabled(self):
        return self._streaming

    def set_system_prompt(self, template=None, variables=None):
        self.system_prompt = template

    async def process_request(self, input_text, user_id, session_id, chat_history, additional_params=None):
        self.received_input = input_text
        self.received_history = chat_history
        if self._streaming:
            async def gen():
                yield AgentStreamResponse(text=self._reply, final_message=_msg(self._reply))
            return gen()
        return _msg(self._reply)


def _tools():
    return AgentTools([AgentTool(name="search_products", func=lambda query: "unused")])


def _build(gatherer, presenter, tools, **kwargs):
    return GroundedAgent(GroundedAgentOptions(
        name="Shop", description="grounded shop", gatherer=gatherer, presenter=presenter,
        tools=tools, **kwargs,
    ))


# ---------------------------------------------------------------------------
# Tool turn -> presenter runs, grounded on the curated feed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_turn_runs_presenter_with_curated_feed():
    tools = _tools()
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[("search_products", {"query": "x"}, "Blue shoes, €90")],
                             draft="brain draft that must not surface")
    presenter = _FakePresenter(AgentOptions(name="p", description=""), reply="Blue shoes cost €90.")
    agent = _build(gatherer, presenter, tools)

    result = await agent.process_request("shoes?", "u", "s", [])

    assert result.content[0]["text"] == "Blue shoes cost €90."
    # Presenter got the tagged question + curated feed, never the raw draft.
    assert "<user question>\nshoes?" in presenter.received_input
    assert "### search_products\nBlue shoes, €90" in presenter.received_input
    assert "brain draft" not in presenter.received_input


@pytest.mark.asyncio
async def test_presenter_prompt_resolved_from_primary_tool():
    tools = _tools()
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[("get_order", {}, "shipped")], draft="")
    presenter = _FakePresenter(AgentOptions(name="p", description=""))
    prompt = PresenterPrompt(default="DEFAULT", per_tool={"get_order": "ORDER PROMPT"})
    agent = _build(gatherer, presenter, tools, presenter_prompt=prompt)

    await agent.process_request("where is my order?", "u", "s", [])
    assert presenter.system_prompt == "ORDER PROMPT"


@pytest.mark.asyncio
async def test_presenter_prompt_falls_back_to_default():
    tools = _tools()
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[("unmapped", {}, "data")], draft="")
    presenter = _FakePresenter(AgentOptions(name="p", description=""))
    agent = _build(gatherer, presenter, tools,
                   presenter_prompt=PresenterPrompt(default="DEFAULT", per_tool={"other": "X"}))
    await agent.process_request("q", "u", "s", [])
    assert presenter.system_prompt == "DEFAULT"


@pytest.mark.asyncio
async def test_history_forwarded_to_presenter():
    tools = _tools()
    history = [ConversationMessage(role=ParticipantRole.USER.value, content=[{"text": "earlier"}])]
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[("t", {}, "d")], draft="")
    presenter = _FakePresenter(AgentOptions(name="p", description=""))
    agent = _build(gatherer, presenter, tools)
    await agent.process_request("q", "u", "s", history)
    assert presenter.received_history is history


# ---------------------------------------------------------------------------
# No-tool turn -> skip presenter, emit the gatherer's own reply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_tool_turn_skips_presenter():
    tools = _tools()
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[], draft="Hi there!")
    presenter = _FakePresenter(AgentOptions(name="p", description=""))
    presenter.process_request = AsyncMock()
    agent = _build(gatherer, presenter, tools)

    result = await agent.process_request("hello", "u", "s", [])
    assert result.content[0]["text"] == "Hi there!"
    presenter.process_request.assert_not_awaited()


# ---------------------------------------------------------------------------
# Capture seam restores the original callbacks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_original_tool_callbacks_restored():
    tools = _tools()
    sentinel = tools.callbacks
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[("t", {}, "d")], draft="")
    presenter = _FakePresenter(AgentOptions(name="p", description=""))
    agent = _build(gatherer, presenter, tools)
    await agent.process_request("q", "u", "s", [])
    assert tools.callbacks is sentinel


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_tool_turn_yields_presenter_chunks():
    tools = _tools()
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[("t", {}, "d")], draft="", streaming=True)
    presenter = _FakePresenter(AgentOptions(name="p", description=""), reply="streamed", streaming=True)
    agent = _build(gatherer, presenter, tools)

    assert agent.is_streaming_enabled() is True
    stream = await agent.process_request("q", "u", "s", [])
    chunks = [c async for c in stream]
    assert chunks[-1].final_message.content[0]["text"] == "streamed"


@pytest.mark.asyncio
async def test_streaming_no_tool_turn_yields_gatherer_draft():
    tools = _tools()
    gatherer = _FakeGatherer(AgentOptions(name="g", description=""), tools=tools,
                             tool_calls=[], draft="just chatting", streaming=True)
    presenter = _FakePresenter(AgentOptions(name="p", description=""), streaming=True)
    presenter.process_request = AsyncMock()
    agent = _build(gatherer, presenter, tools)

    stream = await agent.process_request("hi", "u", "s", [])
    chunks = [c async for c in stream]
    assert chunks[-1].final_message.content[0]["text"] == "just chatting"
    presenter.process_request.assert_not_awaited()


# ---------------------------------------------------------------------------
# Curators
# ---------------------------------------------------------------------------

def test_datablock_curator_string_and_structured():
    curator = DataBlockCurator()
    out = curator.curate([
        CapturedToolResult("search", "raw text"),
        CapturedToolResult("detail", {"price": 90, "name": "shoe"}),
    ])
    assert "### search\nraw text" in out
    assert "### detail\n" in out
    assert '"name": "shoe"' in out  # pretty JSON, sorted keys
    assert "\n\n" in out  # sections joined by blank line


def test_per_tool_curator_routes_and_falls_back():
    curator = PerToolCurator(
        formatters={"search": lambda r: f"CUSTOM:{r.result}"},
    )
    out = curator.curate([
        CapturedToolResult("search", "hits"),
        CapturedToolResult("other", "plain"),
    ])
    assert "CUSTOM:hits" in out
    assert "### other\nplain" in out  # unmapped -> dataBlock fallback


# ---------------------------------------------------------------------------
# PresenterPrompt / Grounding helpers
# ---------------------------------------------------------------------------

def test_presenter_prompt_default():
    assert PresenterPrompt.default().resolve(None) == DEFAULT_PRESENTER_PROMPT
    assert PresenterPrompt.default().resolve("anything") == DEFAULT_PRESENTER_PROMPT


def test_grounding_primary_is_last_call():
    results = [CapturedToolResult("a", 1), CapturedToolResult("b", 2)]
    assert Grounding.primary(results).name == "b"
    assert Grounding.primary([]) is None


def test_grounding_presenter_message_tags():
    out = Grounding.presenter_message("Q", "D")
    assert out == "<user question>\nQ\n</user question>\n<data to present>\nD\n</data to present>"


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_requires_gatherer_and_presenter():
    tools = _tools()
    with pytest.raises(ValueError):
        GroundedAgent(GroundedAgentOptions(name="x", description="", gatherer=None,
                                           presenter=_FakePresenter(AgentOptions(name="p", description="")),
                                           tools=tools))


def test_requires_tools():
    g = _FakeGatherer(AgentOptions(name="g", description=""))
    p = _FakePresenter(AgentOptions(name="p", description=""))
    with pytest.raises(ValueError):
        GroundedAgent(GroundedAgentOptions(name="x", description="", gatherer=g, presenter=p, tools=None))
