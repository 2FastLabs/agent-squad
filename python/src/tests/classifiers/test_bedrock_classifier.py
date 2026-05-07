import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from botocore.exceptions import BotoCoreError, ClientError
from agent_squad.classifiers.bedrock_classifier import BedrockClassifier, BedrockClassifierOptions
from agent_squad.classifiers import ClassifierResult, ClassifierCallbacks
from agent_squad.types import ConversationMessage, BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
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


class TestBedrockClassifierOptions:

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        options = BedrockClassifierOptions()

        assert options.model_id is None
        assert options.region is None
        assert options.inference_config == {}
        assert options.client is None
        assert isinstance(options.callbacks, ClassifierCallbacks)

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        mock_client = Mock()
        inference_config = {
            'maxTokens': 2000,
            'temperature': 0.5,
            'top_p': 0.8,
            'stop_sequences': ['STOP']
        }
        callbacks = ClassifierCallbacks()

        options = BedrockClassifierOptions(
            model_id="custom-model",
            region="us-west-2",
            inference_config=inference_config,
            client=mock_client,
            callbacks=callbacks
        )

        assert options.model_id == "custom-model"
        assert options.region == "us-west-2"
        assert options.inference_config == inference_config
        assert options.client == mock_client
        assert options.callbacks == callbacks


class TestBedrockClassifier:

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.options = BedrockClassifierOptions(client=self.mock_client)
        self.mock_agents = {
            'agent-1': MockAgent('agent-1', 'First test agent'),
            'agent-2': MockAgent('agent-2', 'Second test agent')
        }

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_with_client_injection(self, mock_user_agent):
        """Test initialization with injected client"""
        mock_client = Mock()
        options = BedrockClassifierOptions(client=mock_client)

        classifier = BedrockClassifier(options)

        assert classifier.client == mock_client
        assert classifier.model_id == BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
        assert isinstance(classifier.callbacks, ClassifierCallbacks)
        mock_user_agent.register_feature_to_client.assert_called_once_with(
            mock_client, feature="bedrock-classifier"
        )

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @patch('agent_squad.classifiers.bedrock_classifier.boto3')
    def test_init_boto3_auto_creation(self, mock_boto3, mock_user_agent):
        """Test initialization creates boto3 client when none provided"""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        options = BedrockClassifierOptions(region="us-east-1")

        classifier = BedrockClassifier(options)

        mock_boto3.client.assert_called_once_with('bedrock-runtime', region_name="us-east-1")
        assert classifier.client == mock_client

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_default_model_id(self, mock_user_agent):
        """Test initialization uses default model ID"""
        classifier = BedrockClassifier(self.options)

        assert classifier.model_id == BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_custom_model_id(self, mock_user_agent):
        """Test initialization with custom model ID"""
        options = BedrockClassifierOptions(
            client=Mock(),
            model_id="anthropic.claude-3-haiku-20240307-v1:0"
        )

        classifier = BedrockClassifier(options)

        assert classifier.model_id == "anthropic.claude-3-haiku-20240307-v1:0"

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_default_inference_config(self, mock_user_agent):
        """Test initialization with default inference configuration"""
        classifier = BedrockClassifier(self.options)

        assert classifier.inference_config['maxTokens'] == 1000
        assert classifier.inference_config['temperature'] == 0.0
        assert classifier.inference_config['topP'] == 0.9
        assert classifier.inference_config['stopSequences'] == []

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_custom_inference_config(self, mock_user_agent):
        """Test initialization with custom inference configuration"""
        inference_config = {
            'maxTokens': 2000,
            'temperature': 0.7,
            'top_p': 0.95,
            'stop_sequences': ['END']
        }
        options = BedrockClassifierOptions(
            client=Mock(),
            inference_config=inference_config
        )

        classifier = BedrockClassifier(options)

        assert classifier.inference_config['maxTokens'] == 2000
        assert classifier.inference_config['temperature'] == 0.7
        assert classifier.inference_config['topP'] == 0.95
        assert classifier.inference_config['stopSequences'] == ['END']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_tools_schema_structure(self, mock_user_agent):
        """Test that tools are properly configured with correct schema"""
        classifier = BedrockClassifier(self.options)

        assert len(classifier.tools) == 1
        tool = classifier.tools[0]

        assert 'toolSpec' in tool
        tool_spec = tool['toolSpec']
        assert tool_spec['name'] == 'analyzePrompt'
        assert 'description' in tool_spec
        assert 'inputSchema' in tool_spec

        schema = tool_spec['inputSchema']['json']
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'required' in schema

        properties = schema['properties']
        assert 'userinput' in properties
        assert 'selected_agent' in properties
        assert 'confidence' in properties

        assert schema['required'] == ['userinput', 'selected_agent', 'confidence']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_success(self, mock_user_agent):
        """Test successful request processing with agent selection"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.85
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        chat_history = [
            ConversationMessage(role='user', content=[{'text': 'Previous message'}])
        ]

        result = await classifier.process_request("Test input", chat_history)

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent.id == 'agent-1'
        assert result.confidence == 0.85
        self.mock_client.converse.assert_called_once()

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_no_output(self, mock_user_agent):
        """Test handling when no output is received from Bedrock"""
        mock_response = {}

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No output received from Bedrock model"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_no_tool_use_in_content(self, mock_user_agent):
        """Test handling when response content has no toolUse block"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {'text': 'Regular text response'}
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No valid tool use found in the response"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @patch('agent_squad.classifiers.bedrock_classifier.is_tool_input')
    @pytest.mark.asyncio
    async def test_process_request_invalid_tool_input(self, mock_is_tool_input, mock_user_agent):
        """Test handling when tool input does not match expected structure"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {'invalid': 'data'}
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response
        mock_is_tool_input.return_value = False

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="Tool input does not match expected structure"):
            await classifier.process_request("Test input", [])

        mock_is_tool_input.assert_called_once_with({'invalid': 'data'})

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_empty_tool_use(self, mock_user_agent):
        """Test handling when toolUse is empty/None"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': None
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No tool use found in the response"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_botocore_error(self, mock_user_agent):
        """Test handling of BotoCoreError"""
        self.mock_client.converse.side_effect = BotoCoreError()

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(BotoCoreError):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_client_error(self, mock_user_agent):
        """Test handling of ClientError"""
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid model ID'
            }
        }
        self.mock_client.converse.side_effect = ClientError(error_response, 'Converse')

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ClientError):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_callbacks(self, mock_user_agent):
        """Test process_request triggers on_classifier_start and on_classifier_stop callbacks"""
        mock_callbacks = AsyncMock(spec=ClassifierCallbacks)

        options = BedrockClassifierOptions(
            client=self.mock_client,
            callbacks=mock_callbacks
        )

        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.85
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        mock_callbacks.on_classifier_start.assert_called_once()
        mock_callbacks.on_classifier_stop.assert_called_once()

        start_call = mock_callbacks.on_classifier_start.call_args
        assert start_call[0] == ('on_classifier_start', 'Test input')

        stop_call = mock_callbacks.on_classifier_stop.call_args
        assert stop_call[0][0] == 'on_classifier_stop'
        assert isinstance(stop_call[0][1], ClassifierResult)
        assert 'usage' in stop_call[1]

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_agent_not_found(self, mock_user_agent):
        """Test process_request when selected agent does not exist"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'non-existent-agent',
                                    'confidence': 0.9
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent is None
        assert result.confidence == 0.9

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_confidence_as_float(self, mock_user_agent):
        """Test that confidence is properly converted to float"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': '0.95'
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert isinstance(result.confidence, float)
        assert result.confidence == 0.95

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_tool_choice_for_anthropic_model(self, mock_user_agent):
        """Test that toolChoice is set for anthropic models"""
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.85
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert 'toolChoice' in call_kwargs['toolConfig']
        assert call_kwargs['toolConfig']['toolChoice'] == {
            'tool': {'name': 'analyzePrompt'}
        }

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_no_tool_choice_for_non_anthropic_model(self, mock_user_agent):
        """Test that toolChoice is not set for non-anthropic/non-mistral models"""
        options = BedrockClassifierOptions(
            client=self.mock_client,
            model_id="amazon.titan-text-express-v1"
        )

        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.85
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert 'toolChoice' not in call_kwargs['toolConfig']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_custom_inference_config(self, mock_user_agent):
        """Test process_request uses custom inference configuration in API call"""
        inference_config = {
            'maxTokens': 1500,
            'temperature': 0.5,
            'top_p': 0.8,
            'stop_sequences': ['STOP', 'END']
        }
        options = BedrockClassifierOptions(
            client=self.mock_client,
            inference_config=inference_config
        )

        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.8
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        self.mock_client.converse.return_value = mock_response

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert call_kwargs['inferenceConfig']['maxTokens'] == 1500
        assert call_kwargs['inferenceConfig']['temperature'] == 0.5
        assert call_kwargs['inferenceConfig']['topP'] == 0.8
        assert call_kwargs['inferenceConfig']['stopSequences'] == ['STOP', 'END']
