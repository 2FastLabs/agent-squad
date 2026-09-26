import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncIterable
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents import AgentStreamResponse
from agent_squad.agents.gemini_agent import GeminiAgent, GeminiAgentOptions


@pytest.fixture
def mock_genai_client():
    mock_client = Mock()
    mock_client.models = Mock()
    mock_client.models.generate_content = Mock()
    mock_client.models.generate_content_stream = Mock()
    return mock_client


@pytest.fixture
def gemini_agent(mock_genai_client):
    with patch('google.genai.Client', return_value=mock_genai_client):
        options = GeminiAgentOptions(
            name="TestAgent",
            description="A test Gemini agent",
            api_key="test-api-key",
            model_id="gemini-2.0-flash",
            streaming=False,
            inference_config={
                'maxTokens': 500,
                'temperature': 0.5,
                'topP': 0.8,
                'stopSequences': []
            }
        )
        agent = GeminiAgent(options)
        agent.client = mock_genai_client
        return agent


def test_gemini_agent_requires_api_key_or_client():
    with pytest.raises(ValueError, match="API key or a pre-configured genai.Client is required"):
        GeminiAgentOptions(
            name="TestAgent",
            description="Test",
        )
        GeminiAgent(GeminiAgentOptions(
            name="TestAgent",
            description="Test",
        ))


def test_gemini_agent_client_injection():
    """Passing a pre-configured client bypasses api_key requirement."""
    mock_client = Mock()
    options = GeminiAgentOptions(
        name="TestAgent",
        description="Test",
        client=mock_client,
    )
    agent = GeminiAgent(options)
    assert agent.client is mock_client


def test_custom_system_prompt_with_variable():
    mock_client = Mock()
    options = GeminiAgentOptions(
        name="TestAgent",
        description="A test agent",
        client=mock_client,
        custom_system_prompt={
            'template': "This is a prompt with {{variable}}",
            'variables': {'variable': 'value'}
        }
    )
    agent = GeminiAgent(options)
    assert agent.system_prompt == "This is a prompt with value"


@pytest.mark.asyncio
async def test_process_request_success(gemini_agent, mock_genai_client):
    mock_response = Mock()
    mock_response.text = "Test response"
    mock_genai_client.models.generate_content.return_value = mock_response

    result = await gemini_agent.process_request(
        "Test question",
        "test_user",
        "test_session",
        []
    )

    assert isinstance(result, ConversationMessage)
    assert result.role == ParticipantRole.ASSISTANT.value
    assert result.content[0]['text'] == "Test response"


@pytest.mark.asyncio
async def test_process_request_streaming(gemini_agent, mock_genai_client):
    gemini_agent.streaming = True

    class MockChunk:
        def __init__(self, text):
            self.text = text

    mock_stream = [
        MockChunk("This "),
        MockChunk("is "),
        MockChunk("a "),
        MockChunk("test response"),
    ]
    mock_genai_client.models.generate_content_stream.return_value = mock_stream

    result = await gemini_agent.process_request(
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
            assert chunk.final_message.content[0]['text'] == "This is a test response"
    assert chunks == ["This ", "is ", "a ", "test response"]


@pytest.mark.asyncio
async def test_process_request_with_retriever(gemini_agent, mock_genai_client):
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_and_combine_results.return_value = "Context from retriever"
    gemini_agent.retriever = mock_retriever

    mock_response = Mock()
    mock_response.text = "Response with context"
    mock_genai_client.models.generate_content.return_value = mock_response

    result = await gemini_agent.process_request(
        "Test question",
        "test_user",
        "test_session",
        []
    )

    mock_retriever.retrieve_and_combine_results.assert_called_once_with("Test question")
    assert isinstance(result, ConversationMessage)
    assert result.content[0]['text'] == "Response with context"


@pytest.mark.asyncio
async def test_process_request_api_error(gemini_agent, mock_genai_client):
    mock_genai_client.models.generate_content.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        await gemini_agent.process_request(
            "Test input",
            "user123",
            "session456",
            []
        )
    assert "API Error" in str(exc_info.value)


def test_is_streaming_enabled(gemini_agent):
    assert not gemini_agent.is_streaming_enabled()
    gemini_agent.streaming = True
    assert gemini_agent.is_streaming_enabled()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_vertex_ai_integration():
    """Integration test using Vertex AI with real credentials.

    Run with:
        GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json \
        pytest -m integration python/src/tests/agents/test_gemini_agent.py -v
    """
    from google import genai

    client = genai.Client(
        vertexai=True,
        project="project-cf964f7d-d79b-4b69-81c",
        location="us-central1"
    )
    options = GeminiAgentOptions(
        name="VertexTest",
        description="Integration test agent",
        client=client,
        model_id="gemini-2.0-flash",
    )
    agent = GeminiAgent(options)
    result = await agent.process_request("What is 2 + 2?", "user", "sess", [])
    assert isinstance(result, ConversationMessage)
    assert "4" in result.content[0]["text"]
