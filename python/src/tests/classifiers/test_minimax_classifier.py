import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from agent_squad.classifiers.minimax_classifier import (
    MiniMaxClassifier,
    MiniMaxClassifierOptions,
    MINIMAX_API_BASE_URL,
    MINIMAX_MODEL_ID_M2_7
)
from agent_squad.types import ConversationMessage


@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client


@pytest.fixture
def minimax_classifier(mock_openai_client):
    with patch('agent_squad.classifiers.minimax_classifier.OpenAI', return_value=mock_openai_client):
        options = MiniMaxClassifierOptions(
            api_key="test-api-key",
            model_id="MiniMax-M2.7",
            inference_config={
                'max_tokens': 500,
                'temperature': 0.5,
                'top_p': 0.9
            }
        )
        classifier = MiniMaxClassifier(options)
        classifier.client = mock_openai_client
        return classifier


class TestMiniMaxClassifierInit:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key=None
            ))

    def test_default_model(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI'):
            classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key"
            ))
            assert classifier.model_id == MINIMAX_MODEL_ID_M2_7

    def test_custom_model(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI'):
            classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key",
                model_id="MiniMax-M2.7-highspeed"
            ))
            assert classifier.model_id == "MiniMax-M2.7-highspeed"

    def test_default_base_url(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI') as mock_openai:
            MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key"
            ))
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url=MINIMAX_API_BASE_URL
            )

    def test_custom_base_url(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI') as mock_openai:
            MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key",
                base_url="https://custom.example.com/v1"
            ))
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.example.com/v1"
            )

    def test_temperature_clamping_zero(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI'):
            classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key",
                inference_config={'temperature': 0}
            ))
            assert classifier.inference_config['temperature'] == 0.01

    def test_temperature_clamping_negative(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI'):
            classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key",
                inference_config={'temperature': -1.0}
            ))
            assert classifier.inference_config['temperature'] == 0.01

    def test_temperature_positive_not_clamped(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI'):
            classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key",
                inference_config={'temperature': 0.7}
            ))
            assert classifier.inference_config['temperature'] == 0.7

    def test_default_inference_config(self):
        with patch('agent_squad.classifiers.minimax_classifier.OpenAI'):
            classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
                api_key="test-key"
            ))
            assert classifier.inference_config['max_tokens'] == 1000
            assert classifier.inference_config['temperature'] == 0.1
            assert classifier.inference_config['top_p'] == 0.9


class TestMiniMaxClassifierProcessRequest:
    @pytest.mark.asyncio
    async def test_process_request_success(self, minimax_classifier, mock_openai_client):
        # Set up a mock agent - key must match get_agent_by_id logic (lowercase)
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.id = "testagent"
        minimax_classifier.agents = {"testagent": mock_agent}
        minimax_classifier.agent_descriptions = "testagent: A test agent"

        # Set up mock response
        mock_tool_call = Mock()
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "analyzePrompt"
        mock_tool_call.function.arguments = json.dumps({
            "userinput": "Test input",
            "selected_agent": "testagent",
            "confidence": 0.95
        })

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await minimax_classifier.process_request("Test input", [])

        assert result.selected_agent == mock_agent
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_process_request_api_error(self, minimax_classifier, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            await minimax_classifier.process_request("Test input", [])
        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_request_invalid_tool_call(self, minimax_classifier, mock_openai_client):
        mock_tool_call = Mock()
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "wrongFunction"
        mock_tool_call.function.arguments = json.dumps({})

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="No valid tool call found"):
            await minimax_classifier.process_request("Test input", [])
