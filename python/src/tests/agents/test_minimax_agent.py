import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import AsyncIterable
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents import AgentStreamResponse
from agent_squad.agents.minimax_agent import MiniMaxAgent, MiniMaxAgentOptions, MINIMAX_API_BASE_URL


@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client


@pytest.fixture
def minimax_agent(mock_openai_client):
    with patch('agent_squad.agents.minimax_agent.OpenAI', return_value=mock_openai_client):
        options = MiniMaxAgentOptions(
            name="TestMiniMaxAgent",
            description="A test MiniMax agent",
            api_key="test-api-key",
            model="MiniMax-M2.7",
            streaming=False,
            inference_config={
                'maxTokens': 500,
                'temperature': 0.5,
                'topP': 0.8,
                'stopSequences': []
            }
        )
        agent = MiniMaxAgent(options)
        agent.client = mock_openai_client
        return agent


class TestMiniMaxAgentInit:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key=None
            ))

    def test_default_model(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key"
            ))
            assert agent.model == "MiniMax-M2.7"

    def test_custom_model(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                model="MiniMax-M2.7-highspeed"
            ))
            assert agent.model == "MiniMax-M2.7-highspeed"

    def test_default_base_url(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI') as mock_openai:
            MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key"
            ))
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url=MINIMAX_API_BASE_URL
            )

    def test_custom_base_url(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI') as mock_openai:
            MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                base_url="https://custom.api.example.com/v1"
            ))
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.api.example.com/v1"
            )

    def test_custom_client(self):
        mock_client = Mock()
        with patch('agent_squad.agents.minimax_agent.OpenAI') as mock_openai:
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                client=mock_client
            ))
            assert agent.client is mock_client
            mock_openai.assert_not_called()

    def test_temperature_clamping_zero(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                inference_config={'temperature': 0}
            ))
            assert agent.inference_config['temperature'] == 0.01

    def test_temperature_clamping_negative(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                inference_config={'temperature': -0.5}
            ))
            assert agent.inference_config['temperature'] == 0.01

    def test_temperature_positive_not_clamped(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                inference_config={'temperature': 0.7}
            ))
            assert agent.inference_config['temperature'] == 0.7

    def test_temperature_none_not_clamped(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key"
            ))
            assert agent.inference_config['temperature'] is None

    def test_default_inference_config(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key"
            ))
            assert agent.inference_config['maxTokens'] == 1000
            assert agent.inference_config['temperature'] is None
            assert agent.inference_config['topP'] is None
            assert agent.inference_config['stopSequences'] is None

    def test_custom_inference_config(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key",
                inference_config={
                    'maxTokens': 2000,
                    'temperature': 0.8,
                    'topP': 0.95
                }
            ))
            assert agent.inference_config['maxTokens'] == 2000
            assert agent.inference_config['temperature'] == 0.8
            assert agent.inference_config['topP'] == 0.95


class TestMiniMaxAgentSystemPrompt:
    def test_custom_system_prompt_with_variable(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            options = MiniMaxAgentOptions(
                name="TestAgent",
                description="A test agent",
                api_key="test-api-key",
                custom_system_prompt={
                    'template': "This is a prompt with {{variable}}",
                    'variables': {'variable': 'value'}
                }
            )
            agent = MiniMaxAgent(options)
            assert agent.system_prompt == "This is a prompt with value"

    def test_set_system_prompt(self):
        with patch('agent_squad.agents.minimax_agent.OpenAI'):
            agent = MiniMaxAgent(MiniMaxAgentOptions(
                name="Test",
                description="Test",
                api_key="test-key"
            ))
            agent.set_system_prompt(
                template="Hello {{name}}, you are a {{role}}",
                variables={'name': 'MiniMax', 'role': 'assistant'}
            )
            assert agent.system_prompt == "Hello MiniMax, you are a assistant"

    def test_replace_placeholders_with_list(self):
        result = MiniMaxAgent.replace_placeholders(
            "Items: {{items}}",
            {'items': ['one', 'two', 'three']}
        )
        assert result == "Items: one\ntwo\nthree"

    def test_replace_placeholders_unknown_variable(self):
        result = MiniMaxAgent.replace_placeholders(
            "Hello {{unknown}}",
            {}
        )
        assert result == "Hello {{unknown}}"


class TestMiniMaxAgentProcessRequest:
    @pytest.mark.asyncio
    async def test_process_request_success(self, minimax_agent, mock_openai_client):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a test response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await minimax_agent.process_request(
            "Test question",
            "test_user",
            "test_session",
            []
        )

        assert isinstance(result, ConversationMessage)
        assert result.role == ParticipantRole.ASSISTANT.value
        assert result.content[0]['text'] == 'This is a test response'

    @pytest.mark.asyncio
    async def test_process_request_streaming(self, minimax_agent, mock_openai_client):
        minimax_agent.streaming = True

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

        result = await minimax_agent.process_request(
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
    async def test_process_request_with_retriever(self, minimax_agent, mock_openai_client):
        mock_retriever = AsyncMock()
        mock_retriever.retrieve_and_combine_results.return_value = "Context from retriever"
        minimax_agent.retriever = mock_retriever

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response with context"
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await minimax_agent.process_request(
            "Test question",
            "test_user",
            "test_session",
            []
        )

        mock_retriever.retrieve_and_combine_results.assert_called_once_with("Test question")
        assert isinstance(result, ConversationMessage)
        assert result.content[0]['text'] == "Response with context"

    @pytest.mark.asyncio
    async def test_process_request_api_error(self, minimax_agent, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            await minimax_agent.process_request(
                "Test input",
                "user123",
                "session456",
                []
            )
        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_request_with_chat_history(self, minimax_agent, mock_openai_client):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Follow-up response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        chat_history = [
            ConversationMessage(
                role=ParticipantRole.USER.value,
                content=[{"text": "Previous question"}]
            ),
            ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": "Previous answer"}]
            )
        ]

        result = await minimax_agent.process_request(
            "Follow-up question",
            "test_user",
            "test_session",
            chat_history
        )

        assert isinstance(result, ConversationMessage)
        assert result.content[0]['text'] == "Follow-up response"

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs.get('messages', call_args[1].get('messages', []))
        assert len(messages) == 4  # system + 2 history + user


class TestMiniMaxAgentHandlers:
    @pytest.mark.asyncio
    async def test_handle_single_response_no_choices(self, minimax_agent, mock_openai_client):
        mock_response = Mock()
        mock_response.choices = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match='No choices returned from MiniMax API'):
            await minimax_agent.handle_single_response({
                "model": "MiniMax-M2.7",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False
            })

    @pytest.mark.asyncio
    async def test_handle_single_response_non_string(self, minimax_agent, mock_openai_client):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = 12345  # Not a string
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match='Unexpected response format from MiniMax API'):
            await minimax_agent.handle_single_response({
                "model": "MiniMax-M2.7",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False
            })

    def test_is_streaming_enabled(self, minimax_agent):
        assert not minimax_agent.is_streaming_enabled()
        minimax_agent.streaming = True
        assert minimax_agent.is_streaming_enabled()
