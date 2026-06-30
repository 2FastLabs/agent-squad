"""Unit tests for ChainAgent."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_squad.agents.chain_agent import ChainAgent, ChainAgentOptions
from agent_squad.types import ConversationMessage, ParticipantRole


def _make_agent(name: str, response_text: str = "ok") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.process_request = AsyncMock(return_value=ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": response_text}],
    ))
    return agent


def _options(agents, **kwargs):
    return ChainAgentOptions(
        name="chain",
        description="test chain",
        agents=agents,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_agent_returns_response():
    agent = _make_agent("a", "hello")
    chain = ChainAgent(_options([agent]))
    result = await chain.process_request("hi", "u", "s", [])
    assert result.content[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_output_of_first_agent_fed_to_second():
    a1 = _make_agent("a1", "step1")
    a2 = _make_agent("a2", "step2")
    chain = ChainAgent(_options([a1, a2]))
    await chain.process_request("start", "u", "s", [])
    a2.process_request.assert_awaited_once()
    call_input = a2.process_request.call_args[0][0]
    assert call_input == "step1"


# ---------------------------------------------------------------------------
# Exception handling — the core bug fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_exception_raises_exception_not_string():
    agent = MagicMock()
    agent.name = "bad_agent"
    agent.process_request = AsyncMock(side_effect=RuntimeError("boom"))
    chain = ChainAgent(_options([agent]))

    with pytest.raises(Exception) as exc_info:
        await chain.process_request("hi", "u", "s", [])

    # Must be a real Exception, not a string
    assert isinstance(exc_info.value, Exception)
    assert "bad_agent" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_exception_chained_from_original():
    original = RuntimeError("root cause")
    agent = MagicMock()
    agent.name = "bad_agent"
    agent.process_request = AsyncMock(side_effect=original)
    chain = ChainAgent(_options([agent]))

    with pytest.raises(Exception) as exc_info:
        await chain.process_request("hi", "u", "s", [])

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_agents_raises_value_error():
    with pytest.raises(ValueError):
        ChainAgent(_options([]))


@pytest.mark.asyncio
async def test_default_output_used_when_agent_returns_no_text():
    agent = MagicMock()
    agent.name = "empty"
    agent.process_request = AsyncMock(return_value=ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"no_text": True}],
    ))
    chain = ChainAgent(_options([agent], default_output="fallback"))
    result = await chain.process_request("hi", "u", "s", [])
    assert result.content[0]["text"] == "fallback"
