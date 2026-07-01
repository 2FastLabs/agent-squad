"""The 2-LLM grounded (anti-hallucination) pattern as an ``Agent``.

A *gatherer* LLM calls tools and sees the raw results but never speaks to the user; an isolated
*presenter* LLM writes the reply grounded only on the curated facts — no tools, no tool responses —
so it cannot invent values beyond what was fetched. A chit-chat turn that calls no tools is answered
in one pass by the gatherer, skipping the presenter.

This mirrors the Swift ``GroundedAgent``. The tool-UI (widget) and trace-span machinery from the
Swift implementation has no equivalent in this tree and is intentionally omitted.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Callable, Optional, Union

from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils import AgentToolCallbacks, AgentTools, Logger

from .agent import Agent, AgentOptions, AgentStreamResponse

# The generic grounding instruction: present only the provided data, never invent values.
DEFAULT_PRESENTER_PROMPT = (
    "You are presenting information to the user. Use ONLY the data provided. Be concise and "
    "natural, and never invent or infer values that are not present in the data."
)


@dataclass
class CapturedToolResult:
    """One captured tool call, as the curator sees it."""

    name: str
    result: Any


class ToolOutputCurator(ABC):
    """Turns gathered tool results into the text the presenter is fed — GroundedAgent's data
    extension point; default is ``DataBlockCurator``. A pure synchronous transform by contract, no
    I/O: a curator needing external data pre-fetches it and is constructed with it."""

    @abstractmethod
    def curate(self, results: list[CapturedToolResult]) -> str:
        raise NotImplementedError


class DataBlockCurator(ToolOutputCurator):
    """The default curator: a faithful ``### <toolName>`` section per tool — the raw string result,
    or the structured result pretty-printed as JSON — concatenated across every captured tool."""

    def curate(self, results: list[CapturedToolResult]) -> str:
        return "\n\n".join(self.section(result) for result in results)

    @staticmethod
    def section(result: CapturedToolResult) -> str:
        """One section. Also used as ``PerToolCurator``'s fallback formatter."""
        if isinstance(result.result, str):
            body = result.result
        else:
            body = json.dumps(result.result, indent=2, sort_keys=True, default=str)
        return f"### {result.name}\n{body}"


# Formats one captured tool into its section of the feed.
Formatter = Callable[[CapturedToolResult], str]


class PerToolCurator(ToolOutputCurator):
    """A curator routing each captured tool to its own formatter, keyed by tool name (the key
    ``PresenterPrompt`` uses), with a fallback for unmapped tools. Each formatter trims/compacts its
    tool's section, so this is where you shrink an oversized payload before the presenter sees it."""

    def __init__(self, formatters: dict[str, Formatter], default: Optional[Formatter] = None):
        self.formatters = formatters
        self.fallback = default or DataBlockCurator.section

    def curate(self, results: list[CapturedToolResult]) -> str:
        return "\n\n".join((self.formatters.get(r.name) or self.fallback)(r) for r in results)


class PresenterPrompt:
    """Chooses the presenter's system prompt, keyed by the turn's primary tool. One prompt by
    default; supply a per-tool map for tool-specific presenters."""

    def __init__(self, default: str, per_tool: Optional[dict[str, str]] = None):
        self._default = default
        self._per_tool = per_tool or {}

    def resolve(self, primary_tool: Optional[str]) -> str:
        """The prompt for a turn whose primary tool is ``primary_tool`` (falls back to the default)."""
        if primary_tool and primary_tool in self._per_tool:
            return self._per_tool[primary_tool]
        return self._default

    @classmethod
    def default(cls) -> "PresenterPrompt":
        return cls(default=DEFAULT_PRESENTER_PROMPT)


class Grounding:
    """Shared helpers for the gather -> present pattern."""

    @staticmethod
    def primary(results: list[CapturedToolResult]) -> Optional[CapturedToolResult]:
        """The turn's primary tool: the last call (drives the presenter prompt selection)."""
        return results[-1] if results else None

    @staticmethod
    def presenter_message(question: str, data: str) -> str:
        """The presenter's user message: question and curated data, tagged so the model can tell
        them apart."""
        return (
            "<user question>\n"
            f"{question}\n"
            "</user question>\n"
            "<data to present>\n"
            f"{data}\n"
            "</data to present>"
        )


class _CapturingToolCallbacks(AgentToolCallbacks):
    """Wraps the tools' existing callbacks and records every tool result passing through, so the
    curator can build the presenter feed. One per turn — the capture is per-turn state."""

    def __init__(self, inner: Optional[AgentToolCallbacks]):
        self._inner = inner or AgentToolCallbacks()
        self.records: list[CapturedToolResult] = []

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.on_tool_start(*args, **kwargs)

    async def on_tool_end(self, tool_name, payload_input, output, *args: Any, **kwargs: Any) -> Any:
        self.records.append(CapturedToolResult(name=tool_name, result=output))
        return await self._inner.on_tool_end(tool_name, payload_input, output, *args, **kwargs)

    async def on_tool_error(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.on_tool_error(*args, **kwargs)


@dataclass
class GroundedAgentOptions(AgentOptions):
    """Configuration for a :class:`GroundedAgent`.

    Attributes:
        gatherer: The brain agent — configured with ``tools`` and its own system prompt; runs the
            tool loop and never speaks to the user.
        presenter: The presenter agent — should have no tools; writes the grounded reply. May be a
            cheaper/smaller model.
        tools: The ``AgentTools`` the gatherer uses. GroundedAgent hooks it to capture results.
        curator: Shapes gathered results into the presenter feed. Default: ``DataBlockCurator``.
        presenter_prompt: Per-tool presenter system prompts. Default: a generic grounding prompt.
    """

    gatherer: Agent = None
    presenter: Agent = None
    tools: AgentTools = None
    curator: ToolOutputCurator = field(default_factory=DataBlockCurator)
    presenter_prompt: PresenterPrompt = field(default_factory=PresenterPrompt.default)


class GroundedAgent(Agent):
    """The 2-LLM grounded pattern: a gatherer calls tools, an isolated presenter answers only from
    the curated facts. A no-tool turn is answered by the gatherer directly, skipping the presenter."""

    def __init__(self, options: GroundedAgentOptions):
        super().__init__(options)
        if options.gatherer is None or options.presenter is None:
            raise ValueError("GroundedAgent requires both a gatherer and a presenter agent.")
        if options.tools is None:
            raise ValueError("GroundedAgent requires the AgentTools the gatherer uses.")
        self.gatherer = options.gatherer
        self.presenter = options.presenter
        self.tools = options.tools
        self.curator = options.curator
        self.presenter_prompt = options.presenter_prompt

    def is_streaming_enabled(self) -> bool:
        # The final reply is the presenter's, so streaming tracks the presenter (no-tool turns are
        # re-emitted as a single chunk on the streaming path).
        return self.presenter.is_streaming_enabled()

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, Any]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        if self.is_streaming_enabled():
            return self._stream(input_text, user_id, session_id, chat_history, additional_params)
        return await self._collect(input_text, user_id, session_id, chat_history, additional_params)

    # MARK: - Turn

    async def _gather(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, Any]],
    ) -> tuple[str, list[CapturedToolResult]]:
        """Run the gatherer's tool loop against the capturing callbacks. Its tool calls are recorded;
        its own draft reply is returned (used only if no tools were called)."""
        previous = self.tools.callbacks
        capture = _CapturingToolCallbacks(previous)
        self.tools.callbacks = capture
        try:
            response = await self.gatherer.process_request(
                input_text, user_id, session_id, chat_history, additional_params
            )
            draft = await self._drain(response)
        finally:
            self.tools.callbacks = previous
        return draft, capture.records

    async def _present(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, Any]],
        captured: list[CapturedToolResult],
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        """Curate the facts, select the presenter prompt from the primary tool, and run the
        presenter — which has no tools and speaks only from history + question + feed."""
        feed = self.curator.curate(captured)
        primary = Grounding.primary(captured)
        prompt = self.presenter_prompt.resolve(primary.name if primary else None)
        if hasattr(self.presenter, "set_system_prompt"):
            self.presenter.set_system_prompt(prompt)
        else:
            Logger.warn(
                f"Presenter {self.presenter.name} has no set_system_prompt; PresenterPrompt ignored."
            )
        presenter_input = Grounding.presenter_message(input_text, feed)
        return await self.presenter.process_request(
            presenter_input, user_id, session_id, chat_history, additional_params
        )

    async def _collect(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, Any]],
    ) -> ConversationMessage:
        draft, captured = await self._gather(
            input_text, user_id, session_id, chat_history, additional_params
        )
        # No tools called -> chit-chat: speak the gatherer's own reply, skip the presenter.
        if not captured:
            return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": draft}])
        response = await self._present(
            input_text, user_id, session_id, chat_history, additional_params, captured
        )
        if isinstance(response, ConversationMessage):
            return response
        # Presenter streamed despite non-streaming GroundedAgent: drain to a final message.
        text = await self._drain(response)
        return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": text}])

    async def _stream(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, Any]],
    ) -> AsyncIterable[Any]:
        draft, captured = await self._gather(
            input_text, user_id, session_id, chat_history, additional_params
        )
        # No tools called -> re-emit the gatherer's own reply as a single chunk, skip the presenter.
        if not captured:
            message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": draft}])
            yield AgentStreamResponse(text=draft, final_message=message)
            return
        response = await self._present(
            input_text, user_id, session_id, chat_history, additional_params, captured
        )
        if isinstance(response, ConversationMessage):
            text = response.content[0].get("text", "") if response.content else ""
            yield AgentStreamResponse(text=text, final_message=response)
            return
        async for chunk in response:
            yield chunk

    @staticmethod
    async def _drain(response: Union[ConversationMessage, AsyncIterable[Any]]) -> str:
        """Collect the final text from either a complete message or a stream of stream-responses."""
        if isinstance(response, ConversationMessage):
            return response.content[0].get("text", "") if response.content else ""
        text = ""
        async for chunk in response:
            if isinstance(chunk, AgentStreamResponse) and chunk.final_message:
                content = chunk.final_message.content
                text = content[0].get("text", "") if content else ""
        return text
