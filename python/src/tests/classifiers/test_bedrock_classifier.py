import pytest
from unittest.mock import Mock, patch, AsyncMock
from botocore.exceptions import BotoCoreError, ClientError
from agent_squad.classifiers.bedrock_classifier import BedrockClassifier, BedrockClassifierOptions, BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
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
    def test_init_with_provided_client(self, mock_user_agent):
        """Test initialization with a provided boto3 client"""
        classifier = BedrockClassifier(self.options)

        assert classifier.client == self.mock_client
        assert classifier.model_id == BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
        assert isinstance(classifier.callbacks, ClassifierCallbacks)
        mock_user_agent.register_feature_to_client.assert_called_once_with(
            self.mock_client, feature="bedrock-classifier"
        )

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @patch('agent_squad.classifiers.bedrock_classifier.boto3')
    def test_init_creates_client_when_not_provided(self, mock_boto3, mock_user_agent):
        """Test initialization creates boto3 client when none provided"""
        mock_created_client = Mock()
        mock_boto3.client.return_value = mock_created_client

        options = BedrockClassifierOptions(region="us-east-1")
        classifier = BedrockClassifier(options)

        mock_boto3.client.assert_called_once_with('bedrock-runtime', region_name="us-east-1")
        assert classifier.client == mock_created_client
        assert classifier.region == "us-east-1"

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @patch('agent_squad.classifiers.bedrock_classifier.boto3')
    @patch.dict('os.environ', {'AWS_REGION': 'eu-west-1'})
    def test_init_uses_env_region_as_fallback(self, mock_boto3, mock_user_agent):
        """Test initialization falls back to AWS_REGION env var"""
        options = BedrockClassifierOptions()
        classifier = BedrockClassifier(options)

        assert classifier.region == "eu-west-1"
        mock_boto3.client.assert_called_once_with('bedrock-runtime', region_name="eu-west-1")

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_with_custom_model_id(self, mock_user_agent):
        """Test initialization with custom model ID"""
        options = BedrockClassifierOptions(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            client=self.mock_client
        )

        classifier = BedrockClassifier(options)
        assert classifier.model_id == "anthropic.claude-3-haiku-20240307-v1:0"

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_with_custom_inference_config(self, mock_user_agent):
        """Test initialization with custom inference config"""
        inference_config = {
            'maxTokens': 2000,
            'temperature': 0.7,
            'top_p': 0.95,
            'stop_sequences': ['END']
        }
        options = BedrockClassifierOptions(
            client=self.mock_client,
            inference_config=inference_config
        )

        classifier = BedrockClassifier(options)

        assert classifier.inference_config['maxTokens'] == 2000
        assert classifier.inference_config['temperature'] == 0.7
        assert classifier.inference_config['topP'] == 0.95
        assert classifier.inference_config['stopSequences'] == ['END']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_init_with_partial_inference_config_uses_defaults(self, mock_user_agent):
        """Test initialization with partial inference config uses defaults"""
        inference_config = {'temperature': 0.3}
        options = BedrockClassifierOptions(
            client=self.mock_client,
            inference_config=inference_config
        )

        classifier = BedrockClassifier(options)

        assert classifier.inference_config['maxTokens'] == 1000  # default
        assert classifier.inference_config['temperature'] == 0.3  # custom
        assert classifier.inference_config['topP'] == 0.9  # default
        assert classifier.inference_config['stopSequences'] == []  # default

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    def test_tools_configuration(self, mock_user_agent):
        """Test that tools are properly configured for Bedrock Converse API"""
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
        """Test successful request processing"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
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
            'usage': {
                'inputTokens': 100,
                'outputTokens': 50
            }
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        chat_history = [
            ConversationMessage(role='user', content=[{'text': 'Previous message'}])
        ]

        result = await classifier.process_request("Test input", chat_history)

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent.id == 'agent-1'
        assert result.confidence == 0.85

        # Verify the converse call
        self.mock_client.converse.assert_called_once()
        call_kwargs = self.mock_client.converse.call_args[1]

        assert call_kwargs['modelId'] == BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET
        assert call_kwargs['inferenceConfig']['maxTokens'] == 1000
        assert call_kwargs['inferenceConfig']['temperature'] == 0.0
        assert call_kwargs['inferenceConfig']['topP'] == 0.9
        assert call_kwargs['inferenceConfig']['stopSequences'] == []

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_anthropic_model_adds_tool_choice(self, mock_user_agent):
        """Test that toolChoice is added for Anthropic models"""
        options = BedrockClassifierOptions(
            client=self.mock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0"
        )

        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.9
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert 'toolChoice' in call_kwargs['toolConfig']
        assert call_kwargs['toolConfig']['toolChoice'] == {
            'tool': {'name': 'analyzePrompt'}
        }

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_mistral_large_adds_tool_choice(self, mock_user_agent):
        """Test that toolChoice is added for Mistral Large models"""
        options = BedrockClassifierOptions(
            client=self.mock_client,
            model_id="mistral-large-2402-v1:0"
        )

        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.9
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 80, 'outputTokens': 40}
        }

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert 'toolChoice' in call_kwargs['toolConfig']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_non_anthropic_model_no_tool_choice(self, mock_user_agent):
        """Test that toolChoice is NOT added for non-Anthropic/non-Mistral models"""
        options = BedrockClassifierOptions(
            client=self.mock_client,
            model_id="amazon.titan-text-express-v1"
        )

        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
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
            'usage': {'inputTokens': 80, 'outputTokens': 40}
        }

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert 'toolChoice' not in call_kwargs['toolConfig']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_no_output(self, mock_user_agent):
        """Test handling when response has no output"""
        self.mock_client.converse.return_value = {}

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No output received from Bedrock model"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_no_tool_use_in_response(self, mock_user_agent):
        """Test handling when response has no tool use blocks"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {'text': 'Regular text response without tool use'}
                    ]
                }
            }
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No valid tool use found in the response"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_empty_tool_use(self, mock_user_agent):
        """Test handling when toolUse block is empty/falsy"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {'toolUse': None}
                    ]
                }
            }
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No tool use found in the response"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @patch('agent_squad.classifiers.bedrock_classifier.is_tool_input')
    @pytest.mark.asyncio
    async def test_process_request_invalid_tool_input(self, mock_is_tool_input, mock_user_agent):
        """Test handling when tool input doesn't match expected structure"""
        mock_is_tool_input.return_value = False

        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {'invalid': 'data'}
                            }
                        }
                    ]
                }
            }
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="Tool input does not match expected structure"):
            await classifier.process_request("Test input", [])

        mock_is_tool_input.assert_called_once_with({'invalid': 'data'})

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_no_message_content(self, mock_user_agent):
        """Test handling when response message has no content"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {}
            }
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ValueError, match="No valid tool use found in the response"):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_boto_error(self, mock_user_agent):
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
        self.mock_client.converse.side_effect = ClientError(
            error_response={'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            operation_name='Converse'
        )

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        with pytest.raises(ClientError):
            await classifier.process_request("Test input", [])

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_callbacks(self, mock_user_agent):
        """Test process_request invokes callbacks correctly"""
        mock_callbacks = AsyncMock(spec=ClassifierCallbacks)

        options = BedrockClassifierOptions(
            client=self.mock_client,
            callbacks=mock_callbacks
        )

        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
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

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        # Verify callbacks were called
        mock_callbacks.on_classifier_start.assert_called_once()
        mock_callbacks.on_classifier_stop.assert_called_once()

        # Check start callback arguments
        start_call = mock_callbacks.on_classifier_start.call_args
        assert start_call[0] == ('on_classifier_start', 'Test input')
        assert start_call[1]['modelId'] == BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET

        # Check stop callback arguments
        stop_call = mock_callbacks.on_classifier_stop.call_args
        assert stop_call[0][0] == 'on_classifier_stop'
        assert isinstance(stop_call[0][1], ClassifierResult)
        assert 'usage' in stop_call[1]

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_empty_chat_history(self, mock_user_agent):
        """Test process_request with empty chat history"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-2',
                                    'confidence': 0.75
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 80, 'outputTokens': 40}
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent.id == 'agent-2'
        assert result.confidence == 0.75

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_agent_not_found(self, mock_user_agent):
        """Test process_request when selected agent does not exist"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
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

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result, ClassifierResult)
        assert result.selected_agent is None
        assert result.confidence == 0.9

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_with_custom_inference_config(self, mock_user_agent):
        """Test process_request uses custom inference configuration"""
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

        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
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

        classifier = BedrockClassifier(options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        assert call_kwargs['inferenceConfig']['maxTokens'] == 1500
        assert call_kwargs['inferenceConfig']['temperature'] == 0.5
        assert call_kwargs['inferenceConfig']['topP'] == 0.8
        assert call_kwargs['inferenceConfig']['stopSequences'] == ['STOP', 'END']

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_confidence_as_float(self, mock_user_agent):
        """Test that confidence is properly converted to float"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': '0.95'  # String confidence
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert isinstance(result.confidence, float)
        assert result.confidence == 0.95

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_system_prompt_sent_as_list(self, mock_user_agent):
        """Test that system prompt is passed in Bedrock's expected format"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-1',
                                    'confidence': 0.9
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 100, 'outputTokens': 50}
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        await classifier.process_request("Test input", [])

        call_kwargs = self.mock_client.converse.call_args[1]
        # Bedrock Converse API expects system as a list of text blocks
        assert isinstance(call_kwargs['system'], list)
        assert 'text' in call_kwargs['system'][0]

    @patch('agent_squad.classifiers.bedrock_classifier.user_agent')
    @pytest.mark.asyncio
    async def test_process_request_multiple_content_blocks_finds_tool_use(self, mock_user_agent):
        """Test that tool use is found even among multiple content blocks"""
        self.mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [
                        {'text': 'Some preamble text'},
                        {
                            'toolUse': {
                                'name': 'analyzePrompt',
                                'input': {
                                    'userinput': 'test input',
                                    'selected_agent': 'agent-2',
                                    'confidence': 0.88
                                }
                            }
                        }
                    ]
                }
            },
            'usage': {'inputTokens': 120, 'outputTokens': 60}
        }

        classifier = BedrockClassifier(self.options)
        classifier.set_agents(self.mock_agents)

        result = await classifier.process_request("Test input", [])

        assert result.selected_agent.id == 'agent-2'
        assert result.confidence == 0.88
