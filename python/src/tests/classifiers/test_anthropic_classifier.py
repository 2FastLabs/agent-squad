import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from agent_squad.classifiers.anthropic_classifier import AnthropicClassifier, AnthropicClassifierOptions, ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET
from agent_squad.classifiers import ClassifierResult, ClassifierCallbacks
from agent_squad.types import ConversationMessage
from agent_squad.agents import Agent


class MockAgent(Agent):
    """Mock agent for testing"""
    def __init__(self, agent_id, description="Test agent"):
        super().__init__(type('MockOptions', (), {
            'name': agent_id,
            'description': description,
            'save_chat': True,
            'callbacks': None,
            'LOG_AGENT_DEBUG_TRACE': False
        })())
        self.id = agent_id
        self.description = description

    async def process_request(self, input_text, user_id, session_id, chat_history, additional_params=None):
        return ConversationMessage(role="assistant", content=[{"text": f"Response from {self.id}"}])


class TestAnthropicClassifierOptions:

    def test_init_with_required_params(self):
        """Test initialization with required parameters"""
        options = AnthropicClassifierOptions(api_key="test-api-key")

        assert options.api_key == "test-api-key"
        assert options.model_id is None
        assert options.inference_config == {}
        assert isinstance(options.callbacks, ClassifierCallbacks)

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        inference_config = {
            'max_tokens': 2000,
            'temperature': 0.5,
            'top_p': 0.8,
            'stop_sequences': ['STOP']
        }
        callbacks = ClassifierCallbacks()

        options = AnthropicClassifierOptions(
            api_key="test-api-key",
            model_id="custom-model",
            inference_config=inference_config,
            callbacks=callbacks
        )

        assert options.api_key == "test-api-key"
        assert options.model_id == "custom-model"
        assert options.inference_config == inference_config
        assert options.callbacks == callbacks


class TestAnthropicClassifier:

    def setup_method(self):
        """Set up test fixtures"""
        self.options = AnthropicClassifierOptions(api_key="test-api-key")
        self.mock_agents = {
            'agent-1': MockAgent('agent-1', 'First test agent'),
            'agent-2': MockAgent('agent-2', 'Second test agent')
        }

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_init_with_valid_api_key(self, mock_anthropic):
        """Test initialization with valid API key"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)

        assert classifier.client == mock_client
        assert classifier.model_id == ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET
        assert isinstance(classifier.callbacks, ClassifierCallbacks)
        mock_anthropic.assert_called_once_with(api_key="test-api-key")

    def test_init_without_api_key(self):
        """Test initialization without API key raises ValueError"""
        options = AnthropicClassifierOptions(api_key="")

        with pytest.raises(ValueError, match="Anthropic API key is required"):
            AnthropicClassifier(options)

    def test_init_with_none_api_key(self):
        """Test initialization with None API key raises ValueError"""
        options = AnthropicClassifierOptions(api_key=None)

        with pytest.raises(ValueError, match="Anthropic API key is required"):
            AnthropicClassifier(options)

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_init_with_custom_model_id(self, mock_anthropic):
        """Test initialization with custom model ID"""
        options = AnthropicClassifierOptions(
            api_key="test-api-key",
            model_id="custom-claude-model"
        )

        classifier = AnthropicClassifier(options)
        assert classifier.model_id == "custom-claude-model"

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_init_with_custom_inference_config(self, mock_anthropic):
        """Test initialization with custom inference config"""
        inference_config = {
            'max_tokens': 2000,
            'temperature': 0.7,
            'top_p': 0.95,
            'stop_sequences': ['END']
        }
        options = AnthropicClassifierOptions(
            api_key="test-api-key",
            inference_config=inference_config
        )

        classifier = AnthropicClassifier(options)

        assert classifier.inference_config['max_tokens'] == 2000
        assert classifier.inference_config['temperature'] == 0.7
        assert classifier.inference_config['top_p'] == 0.95
        assert classifier.inference_config['stop_sequences'] == ['END']

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_init_with_partial_inference_config(self, mock_anthropic):
        """Test initialization with partial inference config uses defaults"""
        inference_config = {'temperature': 0.3}
        options = AnthropicClassifierOptions(
            api_key="test-api-key",
            inference_config=inference_config
        )

        classifier = AnthropicClassifier(options)

        assert classifier.inference_config['max_tokens'] == 1000  # default
        assert classifier.inference_config['temperature'] == 0.3  # custom
        assert classifier.inference_config['top_p'] == 0.9  # default
        assert classifier.inference_config['stop_sequences'] == []  # default

    @staticmethod
    def _sent_kwargs(mock_anthropic, inference_config):
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {"selected_agent": "agent-1", "confidence": 0.9}
        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(
            AnthropicClassifierOptions(api_key="test-api-key", inference_config=inference_config)
        )
        classifier.set_agents({'agent-1': Mock(id='agent-1', name='A', description='d')})
        asyncio.run(classifier.process_request("Test input", []))
        return mock_client.messages.create.call_args[1]

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_none_inference_values_are_left_out(self, mock_anthropic):
        kwargs = self._sent_kwargs(mock_anthropic, {'temperature': None, 'top_p': None})
        assert 'temperature' not in kwargs
        assert 'top_p' not in kwargs

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_only_temperature_set(self, mock_anthropic):
        kwargs = self._sent_kwargs(mock_anthropic, {'temperature': 0.3, 'top_p': None})
        assert kwargs['temperature'] == 0.3
        assert 'top_p' not in kwargs

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_only_top_p_set(self, mock_anthropic):
        kwargs = self._sent_kwargs(mock_anthropic, {'temperature': None, 'top_p': 0.8})
        assert 'temperature' not in kwargs
        assert kwargs['top_p'] == 0.8

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    def test_tools_configuration(self, mock_anthropic):
        """Test that tools are properly configured"""
        classifier = AnthropicClassifier(self.options)

        assert len(classifier.tools) == 1
        tool = classifier.tools[0]

        assert tool['name'] == 'analyzePrompt'
        assert 'description' in tool
        assert 'input_schema' in tool

        schema = tool['input_schema']
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'required' in schema

        properties = schema['properties']
        assert 'userinput' in properties
        assert 'selected_agent' in properties
        assert 'confidence' in properties

        assert schema['required'] == ['userinput', 'selected_agent', 'confidence']

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_success(self, mock_anthropic):
        """Test successful request processing"""
        # Mock the Anthropic client response
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {
            'userinput': 'test input',
            'selected_agent': 'agent-1',
            'confidence': 0.85
        }

        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        chat_history = [
            ConversationMessage(role='user', content=[{'text': 'Previous message'}])
        ]

        result = await classifier.process_request("Test input", chat_history)

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent.id == 'agent-1'
        assert result.confidence == 0.85

        # Verify the API call
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]

        assert call_kwargs['model'] == ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET
        assert call_kwargs['max_tokens'] == 1000
        assert call_kwargs['temperature'] == 0.0
        assert call_kwargs['top_p'] == 0.9
        assert call_kwargs['messages'] == [{"role": "user", "content": "Test input"}]
        assert call_kwargs['system'] == "You are an AI assistant."
        assert call_kwargs['tools'] == classifier.tools

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_no_tool_use(self, mock_anthropic):
        """Test handling when no tool use is found in response"""
        mock_text_content = Mock()
        mock_text_content.type = "text"
        mock_text_content.text = "Regular text response"

        mock_response = Mock()
        mock_response.content = [mock_text_content]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No tool use found in the response"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @patch('agent_squad.classifiers.anthropic_classifier.is_tool_input')
    @pytest.mark.asyncio
    async def test_process_request_invalid_tool_input(self, mock_is_tool_input, mock_anthropic):
        """Test handling when tool input is invalid"""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {'invalid': 'data'}

        mock_response = Mock()
        mock_response.content = [mock_tool_use]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Mock is_tool_input to return False
        mock_is_tool_input.return_value = False

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="Tool input does not match expected structure"):
            await classifier.process_request("Test input", [])

        mock_is_tool_input.assert_called_once_with({'invalid': 'data'})

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_api_exception(self, mock_anthropic):
        """Test handling when Anthropic API raises exception"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(Exception, match="API Error"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_with_callbacks(self, mock_anthropic):
        """Test process_request with callbacks"""
        # Mock callbacks
        mock_callbacks = AsyncMock(spec=ClassifierCallbacks)

        options = AnthropicClassifierOptions(
            api_key="test-api-key",
            callbacks=mock_callbacks
        )

        # Mock the Anthropic client response
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {
            'userinput': 'test input',
            'selected_agent': 'agent-1',
            'confidence': 0.85
        }

        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        # Verify callbacks were called
        mock_callbacks.on_classifier_start.assert_called_once()
        mock_callbacks.on_classifier_stop.assert_called_once()

        # Check callback arguments
        start_call = mock_callbacks.on_classifier_start.call_args
        assert start_call[0] == ('on_classifier_start', 'Test input')

        stop_call = mock_callbacks.on_classifier_stop.call_args
        assert stop_call[0][0] == 'on_classifier_stop'
        assert isinstance(stop_call[0][1], ClassifierResult)
        assert 'usage' in stop_call[1]

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_with_empty_chat_history(self, mock_anthropic):
        """Test process_request with empty chat history"""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {
            'userinput': 'test input',
            'selected_agent': 'agent-2',
            'confidence': 0.75
        }

        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 80
        mock_response.usage.output_tokens = 40

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent.id == 'agent-2'
        assert result.confidence == 0.75

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_agent_not_found(self, mock_anthropic):
        """Test process_request when selected agent is not found"""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {
            'userinput': 'test input',
            'selected_agent': 'non-existent-agent',
            'confidence': 0.9
        }

        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent is None  # get_agent_by_id returns None for non-existent agent
        assert result.confidence == 0.9

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_with_custom_inference_config(self, mock_anthropic):
        """Test process_request uses custom inference configuration"""
        inference_config = {
            'max_tokens': 1500,
            'temperature': 0.5,
            'top_p': 0.8,
            'stop_sequences': ['STOP', 'END']
        }

        options = AnthropicClassifierOptions(
            api_key="test-api-key",
            inference_config=inference_config
        )

        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {
            'userinput': 'test input',
            'selected_agent': 'agent-1',
            'confidence': 0.8
        }

        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        # Verify custom inference config was used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs['max_tokens'] == 1500
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['top_p'] == 0.8

    @patch('agent_squad.classifiers.anthropic_classifier.Anthropic')
    @pytest.mark.asyncio
    async def test_process_request_confidence_as_float(self, mock_anthropic):
        """Test that confidence is properly converted to float"""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.input = {
            'userinput': 'test input',
            'selected_agent': 'agent-1',
            'confidence': '0.95'  # String confidence value
        }

        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = AnthropicClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert isinstance(result.confidence, float)
        assert result.confidence == 0.95
