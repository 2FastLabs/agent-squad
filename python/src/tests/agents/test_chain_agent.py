import pytest
from typing import AsyncIterable, Any, Optional
from agent_squad.agents import Agent, AgentOptions, AgentStreamResponse
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents.chain_agent import ChainAgent, ChainAgentOptions


class MockAgent(Agent):
    """Lightweight mock agent with no external dependencies."""
    def __init__(self, name, response=None, side_effect=None):
        super().__init__(AgentOptions(name=name, description=f"Mock {name}"))
        self._response = response
        self._side_effect = side_effect

    async def process_request(self, input_text, *args, **kwargs):
        if self._side_effect:
            raise self._side_effect
        if callable(self._response):
            return self._response(input_text)
        return self._response


class MockStreamingAgent(Agent):
    """Mock agent that returns an async iterable."""
    def __init__(self, name, chunks=None):
        super().__init__(AgentOptions(name=name, description=f"Streaming {name}"))
        self._chunks = chunks or ["chunk1", "chunk2"]
        self.streaming = True

    def is_streaming_enabled(self):
        return True

    async def process_request(self, input_text, *args, **kwargs):
        async def _stream():
            for chunk in self._chunks:
                yield AgentStreamResponse(text=chunk)
            yield AgentStreamResponse(final_message=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": "".join(self._chunks)}]
            ))
        return _stream()


def _make_response(text):
    return ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": text}]
    )


def test_chain_agent_requires_at_least_one_agent():
    with pytest.raises(ValueError, match="at least one agent"):
        ChainAgent(ChainAgentOptions(
            name="Chain", description="Test chain", agents=[]
        ))


@pytest.mark.asyncio
async def test_chain_agent_single_agent_success():
    agent = MockAgent("A", response=_make_response("Hello from A"))
    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain", agents=[agent]
    ))
    result = await chain.process_request("input", "user", "sess", [])
    assert isinstance(result, ConversationMessage)
    assert result.content[0]["text"] == "Hello from A"


@pytest.mark.asyncio
async def test_chain_agent_multi_agent_chaining():
    """Output of agent N becomes input to agent N+1."""
    def echo_upper(input_text):
        return _make_response(input_text.upper())

    agent1 = MockAgent("A", response=_make_response("intermediate"))
    agent2 = MockAgent("B", response=echo_upper)

    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain", agents=[agent1, agent2]
    ))
    result = await chain.process_request("start", "user", "sess", [])
    assert result.content[0]["text"] == "INTERMEDIATE"


@pytest.mark.asyncio
async def test_chain_agent_raises_exception_not_type_error():
    """THE BUG FIX TEST: a failing agent should raise Exception, not TypeError."""
    failing_agent = MockAgent("Fail", side_effect=RuntimeError("boom"))
    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain", agents=[failing_agent]
    ))
    with pytest.raises(Exception, match="Error processing request with agent Fail"):
        await chain.process_request("input", "user", "sess", [])

    # Must NOT be a TypeError (the old bug raised a string literal)
    try:
        await chain.process_request("input", "user", "sess", [])
    except TypeError:
        pytest.fail("Should raise Exception, not TypeError from raising a string literal")
    except Exception:
        pass  # Expected


@pytest.mark.asyncio
async def test_chain_agent_error_propagation():
    """Error in 2nd agent references 2nd agent's name."""
    agent1 = MockAgent("First", response=_make_response("ok"))
    agent2 = MockAgent("Second", side_effect=ValueError("bad value"))

    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain", agents=[agent1, agent2]
    ))
    with pytest.raises(Exception, match="Second"):
        await chain.process_request("input", "user", "sess", [])


@pytest.mark.asyncio
async def test_chain_agent_default_response_on_empty_content():
    """Agent returning empty content triggers default response."""
    empty_agent = MockAgent("Empty", response=ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{}]  # no 'text' key
    ))
    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain", agents=[empty_agent]
    ))
    result = await chain.process_request("input", "user", "sess", [])
    assert result.content[0]["text"] == "No output generated from the chain."


@pytest.mark.asyncio
async def test_chain_agent_streaming_last_agent():
    """Last agent returning async iterable passes through."""
    agent1 = MockAgent("A", response=_make_response("step1"))
    agent2 = MockStreamingAgent("B")

    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain", agents=[agent1, agent2]
    ))
    result = await chain.process_request("input", "user", "sess", [])
    assert hasattr(result, '__aiter__')


@pytest.mark.asyncio
async def test_chain_agent_streaming_intermediate_blocked():
    """Intermediate streaming agent triggers default response."""
    streaming_agent = MockStreamingAgent("Stream")
    normal_agent = MockAgent("Normal", response=_make_response("done"))

    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain",
        agents=[streaming_agent, normal_agent]
    ))
    result = await chain.process_request("input", "user", "sess", [])
    assert result.content[0]["text"] == "No output generated from the chain."


@pytest.mark.asyncio
async def test_chain_agent_custom_default_output():
    """Custom default_output text propagates."""
    empty_agent = MockAgent("Empty", response=ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{}]
    ))
    chain = ChainAgent(ChainAgentOptions(
        name="Chain", description="Test chain",
        agents=[empty_agent],
        default_output="Custom fallback"
    ))
    result = await chain.process_request("input", "user", "sess", [])
    assert result.content[0]["text"] == "Custom fallback"
