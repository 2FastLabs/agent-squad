import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncIterable
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents import OpenAIAgent, OpenAIAgentOptions, AgentStreamResponse

@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    # Set up nested structure to match OpenAI client
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client


@pytest.fixture
def openai_agent(mock_openai_client):
    with patch('openai.OpenAI', return_value=mock_openai_client):
        options = OpenAIAgentOptions(
            name="TestAgent",
            description="A test OpenAI agent",
            api_key="test-api-key",
            model="gpt-4",
            streaming=False,
            inference_config={
                'maxTokens': 500,
                'temperature': 0.5,
                'topP': 0.8,
                'stopSequences': []
            }
        )
        agent = OpenAIAgent(options)
        agent.client = mock_openai_client  # Explicitly set the mock client
        return agent


def test_custom_system_prompt_with_variable():
    with patch('openai.OpenAI'):
        options = OpenAIAgentOptions(
            name="TestAgent",
            description="A test agent",
            api_key="test-api-key",
            custom_system_prompt={
                'template': "This is a prompt with {{variable}}",
                'variables': {'variable': 'value'}
            }
        )
        agent = OpenAIAgent(options)
        assert agent.system_prompt == "This is a prompt with value"


@pytest.mark.asyncio
async def test_process_request_success(openai_agent, mock_openai_client):
    # Create a mock response object
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a test response"
    mock_openai_client.chat.completions.create.return_value = mock_response

    result = await openai_agent.process_request(
        "Test question",
        "test_user",
        "test_session",
        []
    )

    assert isinstance(result, ConversationMessage)
    assert result.role == ParticipantRole.ASSISTANT.value
    assert result.content[0]['text'] == 'This is a test response'


@pytest.mark.asyncio
async def test_process_request_streaming(openai_agent, mock_openai_client):
    openai_agent.streaming = True

    # Create mock chunks
    class MockChunk:
        def __init__(self, content):
            self.choices = [Mock()]
            self.choices[0].delta = Mock()
            self.choices[0].delta.content = content

    mock_stream = [
        MockChunk("This "),
        MockChunk("is "),
        MockChunk("a "),
        MockChunk("test response")
    ]
    mock_openai_client.chat.completions.create.return_value = mock_stream

    result:AgentStreamResponse = await openai_agent.process_request(
        "Test question",
        "test_user",
        "test_session",
        []
    )


    assert isinstance(result, AsyncIterable)
    chunks = []
    async for chunk in result:
        assert isinstance(chunk, AgentStreamResponse)
        if chunk.text:
            chunks.append(chunk.text)
        elif chunk.final_message:
            assert chunk.final_message.role == ParticipantRole.ASSISTANT.value
            assert chunk.final_message.content[0]['text'] == 'This is a test response'
    assert chunks == ["This ", "is ", "a ", "test response"]



@pytest.mark.asyncio
async def test_process_request_with_retriever(openai_agent, mock_openai_client):
    # Set up mock retriever
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_and_combine_results.return_value = "Context from retriever"
    openai_agent.retriever = mock_retriever

    # Set up mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Response with context"
    mock_openai_client.chat.completions.create.return_value = mock_response

    result = await openai_agent.process_request(
        "Test question",
        "test_user",
        "test_session",
        []
    )

    mock_retriever.retrieve_and_combine_results.assert_called_once_with("Test question")
    assert isinstance(result, ConversationMessage)
    assert result.content[0]['text'] == "Response with context"


@pytest.mark.asyncio
async def test_process_request_api_error(openai_agent, mock_openai_client):
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        await openai_agent.process_request(
            "Test input",
            "user123",
            "session456",
            []
        )
    assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_handle_single_response_no_choices(openai_agent, mock_openai_client):
    # Create mock response with no choices
    mock_response = Mock()
    mock_response.choices = []
    mock_openai_client.chat.completions.create.return_value = mock_response

    with pytest.raises(ValueError, match='No choices returned from OpenAI API'):
        await openai_agent.handle_single_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        })


def test_is_streaming_enabled(openai_agent):
    assert not openai_agent.is_streaming_enabled()
    openai_agent.streaming = True
    assert openai_agent.is_streaming_enabled()



async def _count_ticks_while(coro, tick_seconds=0.02):
    """Run coro and, at the same time, a ticker. Returns (result, ticks).
    If coro blocks the event loop, the ticker can't run and ticks stays near 0."""
    ticks = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(tick_seconds)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    try:
        result = await coro
    finally:
        done.set()
        await ticker_task
    return result, ticks


@pytest.mark.asyncio
async def test_single_response_does_not_block_event_loop(openai_agent, mock_openai_client):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "slow answer"

    def slow_create(**kwargs):
        time.sleep(0.3)  # a slow network call on the sync client
        return mock_response

    mock_openai_client.chat.completions.create.side_effect = slow_create

    result, ticks = await _count_ticks_while(
        openai_agent.process_request("Test question", "test_user", "test_session", [])
    )

    assert result.content[0]['text'] == "slow answer"
    assert ticks >= 5  # the event loop kept running during the call


@pytest.mark.asyncio
async def test_streaming_response_does_not_block_event_loop(openai_agent, mock_openai_client):
    openai_agent.streaming = True

    def chunk(content):
        c = Mock()
        c.choices = [Mock()]
        c.choices[0].delta.content = content
        return c

    def slow_stream():
        for content in ["slow ", "stream"]:
            time.sleep(0.15)  # each chunk waits on the network
            yield chunk(content)

    mock_openai_client.chat.completions.create.return_value = slow_stream()

    async def read_all():
        stream = await openai_agent.process_request("Test question", "test_user", "test_session", [])
        return [c.text async for c in stream if c.text]

    texts, ticks = await _count_ticks_while(read_all())

    assert texts == ["slow ", "stream"]
    assert ticks >= 5  # the event loop kept running between chunks
