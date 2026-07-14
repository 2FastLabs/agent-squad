import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents import AnthropicAgent, AnthropicAgentOptions
from agent_squad.utils import Logger, AgentTools, AgentTool
from agent_squad.retrievers import Retriever
from anthropic import Anthropic, AsyncAnthropic
from agent_squad.types import AgentProviderType

logger = Logger()

@pytest.fixture
def mock_anthropic():
    with patch('agent_squad.agents.anthropic_agent.AnthropicAgentOptions.client') as mock:
        yield mock

# Existing tests

def test_no_api_key_init(mock_anthropic):
    try:
        options = AnthropicAgentOptions(
            name="TestAgent",
            description="A test agent",
        )

        _anthropic_llm_agent = AnthropicAgent(options)
        assert False, "Should have raised an exception"
    except Exception as e:
        assert(str(e) == "Anthropic API key or Anthropic client is required")

def test_callbacks_initialization():
    mock_callbacks = MagicMock()

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        callbacks=mock_callbacks
    )

    agent = AnthropicAgent(options)
    assert agent.callbacks is mock_callbacks

def test_client(mock_anthropic):
    try:
        options = AnthropicAgentOptions(
            name="TestAgent",
            description="A test agent",
            client=Anthropic(),
            streaming=True

        )

        _anthropic_llm_agent = AnthropicAgent(options)
        assert False, "Should have raised an exception"
    except Exception as e:
        assert(str(e) == "If streaming is enabled, the provided client must be an AsyncAnthropic client")

    try:
        options = AnthropicAgentOptions(
            name="TestAgent",
            description="A test agent",
            client=AsyncAnthropic(),
            streaming=False

        )

        _anthropic_llm_agent = AnthropicAgent(options)
        assert False, "Should have raised an exception"
    except Exception as e:
        assert(str(e) == "If streaming is disabled, the provided client must be an Anthropic client")

    options = AnthropicAgentOptions(
        name="TestAgent",
        description="A test agent",
        client=AsyncAnthropic(),
        streaming=True

    )
    _anthropic_llm_agent = AnthropicAgent(options)
    assert _anthropic_llm_agent.client is not None


def test_inference_config(mock_anthropic):

    options = AnthropicAgentOptions(
            name="TestAgent",
            description="A test agent",
            client=Anthropic(),
            streaming=False,
            inference_config={
                'temperature': 0.5,
                'topP': 0.5,
                'topK': 0.5,
                'maxTokens': 1000,
            }
        )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert _anthropic_llm_agent.inference_config == {
                'temperature': 0.5,
                'topP': 0.5,
                'topK': 0.5,
                'maxTokens': 1000,
                'stopSequences': []
            }

    options = AnthropicAgentOptions(
            name="TestAgent",
            description="A test agent",
            client=Anthropic(),
            streaming=False,
            inference_config={
                'temperature': 0.5,
                'topK': 0.5,
                'maxTokens': 1000,
            }
        )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert _anthropic_llm_agent.inference_config == {
                'temperature': 0.5,
                'topP': 0.9,
                'topK': 0.5,
                'maxTokens': 1000,
                'stopSequences': []
            }



def test_custom_system_prompt_with_variable(mock_anthropic):
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        }
    )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert(_anthropic_llm_agent.system_prompt == 'This is my new prompt with this value')

def test_custom_system_prompt_with_wrong_variable(mock_anthropic):
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variableT': 'value'}
        }
    )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert(_anthropic_llm_agent.system_prompt == 'This is my new prompt with this {{variable}}')

@pytest.mark.asyncio
async def test_process_request_single_response():
    # Create a mock Anthropic client
    with patch('anthropic.Anthropic') as MockAnthropic:
        # Setup the mock instance that will be created
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = MagicMock(content=[MagicMock(text="Test response")])
        MockAnthropic.return_value = mock_instance

        options = AnthropicAgentOptions(
            name="TestAgent",
            description="A test agent",
            api_key='test-api-key',
            model_id="claude-3-sonnet-20240229",
        )

        anthropic_llm_agent = AnthropicAgent(options)
        anthropic_llm_agent.client = mock_instance # mocking client

        response = await anthropic_llm_agent.process_request('Test prompt', 'user', 'session', [], {})

        # Verify the mock was called
        mock_instance.messages.create.assert_called_once_with(
            model='claude-3-sonnet-20240229',
            max_tokens=1000,
            messages=[{'role': 'user', 'content': 'Test prompt'}],
            system="You are a TestAgent.\n        A test agent\n        Provide helpful and accurate information based on your expertise.\n        You will engage in an open-ended conversation,\n        providing helpful and accurate information based on your expertise.\n        The conversation will proceed as follows:\n        - The human may ask an initial question or provide a prompt on any topic.\n        - You will provide a relevant and informative response.\n        - The human may then follow up with additional questions or prompts related to your previous\n        response, allowing for a multi-turn dialogue on that topic.\n        - Or, the human may switch to a completely new and unrelated topic at any point.\n        - You will seamlessly shift your focus to the new topic, providing thoughtful and\n        coherent responses based on your broad knowledge base.\n        Throughout the conversation, you should aim to:\n        - Understand the context and intent behind each new question or prompt.\n        - Provide substantive and well-reasoned responses that directly address the query.\n        - Draw insights and connections from your extensive knowledge when appropriate.\n        - Ask for clarification if any part of the question or prompt is ambiguous.\n        - Maintain a consistent, respectful, and engaging tone tailored\n        to the human's communication style.\n        - Seamlessly transition between topics as the human introduces new subjects.",
            temperature=0.1,
            top_p=0.9,
            stop_sequences=[]
        )
        assert isinstance(response, ConversationMessage)
        # Fix the assertion to handle MagicMock objects properly
        if hasattr(response.content[0], 'text'):
            assert response.content[0].text == "Test response"
        else:
            assert response.content[0].get('text') == "Test response"
        assert response.role == ParticipantRole.ASSISTANT.value


def test_streaming(mock_anthropic):
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        },
        streaming=True
    )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert(_anthropic_llm_agent.is_streaming_enabled() == True)

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        },
        streaming=False
    )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert(_anthropic_llm_agent.is_streaming_enabled() == False)

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        }
    )

    _anthropic_llm_agent = AnthropicAgent(options)
    assert(_anthropic_llm_agent.is_streaming_enabled() == False)

# New tests to improve coverage

@pytest.mark.asyncio
async def test_prepare_system_prompt_with_retriever():
    # Mock retriever
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve_and_combine_results = AsyncMock(return_value="Retrieved context")

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        retriever=mock_retriever
    )

    anthropic_agent = AnthropicAgent(options)
    system_prompt = await anthropic_agent._prepare_system_prompt("Test query")

    mock_retriever.retrieve_and_combine_results.assert_called_once_with("Test query")
    assert "Retrieved context" in system_prompt

@pytest.mark.asyncio
async def test_prepare_conversation():
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent"
    )

    anthropic_agent = AnthropicAgent(options)

    # Test with empty history
    messages = anthropic_agent._prepare_conversation("New message", [])
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "New message"}

    # Test with conversation history
    history = [
        ConversationMessage(role=ParticipantRole.USER.value, content=[{"text": "User message"}]),
        ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": "Assistant response"}])
    ]

    messages = anthropic_agent._prepare_conversation("New message", history)
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "User message"}
    assert messages[1] == {"role": "assistant", "content": "Assistant response"}
    assert messages[2] == {"role": "user", "content": "New message"}

@pytest.mark.asyncio
async def test_prepare_tool_config():
    # Test with AgentTools
    mock_agent_tools = MagicMock(spec=AgentTools)
    claude_format = {"tools": [{"type": "function", "function": {"name": "test_function"}}]}
    mock_agent_tools.to_claude_format.return_value = claude_format

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        tool_config={"tool": mock_agent_tools}
    )

    anthropic_agent = AnthropicAgent(options)
    result = anthropic_agent._prepare_tool_config()

    mock_agent_tools.to_claude_format.assert_called_once()
    assert result == claude_format

    # Test with list of AgentTool
    mock_agent_tool1 = MagicMock(spec=AgentTool)
    mock_agent_tool1.to_claude_format.return_value = {"type": "function", "function": {"name": "function1"}}

    mock_agent_tool2 = MagicMock(spec=AgentTool)
    mock_agent_tool2.to_claude_format.return_value = {"type": "function", "function": {"name": "function2"}}

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        tool_config={"tool": [mock_agent_tool1, mock_agent_tool2]}
    )

    anthropic_agent = AnthropicAgent(options)
    result = anthropic_agent._prepare_tool_config()

    assert len(result) == 2
    assert result[0] == {"type": "function", "function": {"name": "function1"}}
    assert result[1] == {"type": "function", "function": {"name": "function2"}}

    # Test with invalid tool config
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        tool_config={"tool": "invalid"}
    )

    anthropic_agent = AnthropicAgent(options)
    with pytest.raises(RuntimeError, match="Invalid tool config"):
        anthropic_agent._prepare_tool_config()

def test_build_input():
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent"
    )

    anthropic_agent = AnthropicAgent(options)

    messages = [{"role": "user", "content": "Test message"}]
    system_prompt = "Test system prompt"

    # Test without tools
    input_data = anthropic_agent._build_input(messages, system_prompt)

    assert input_data["model"] == "claude-3-5-sonnet-20240620"
    assert input_data["max_tokens"] == 1000
    assert input_data["messages"] == messages
    assert input_data["system"] == system_prompt
    assert input_data["temperature"] == 0.1
    assert input_data["top_p"] == 0.9
    assert input_data["stop_sequences"] == []
    assert "tools" not in input_data

    # Test with tools
    mock_agent_tools = MagicMock(spec=AgentTools)
    claude_format = {"tools": [{"type": "function", "function": {"name": "test_function"}}]}
    mock_agent_tools.to_claude_format.return_value = claude_format

    anthropic_agent.tool_config = {"tool": mock_agent_tools}

    # Mock _prepare_tool_config to return the claude_format
    anthropic_agent._prepare_tool_config = MagicMock(return_value=claude_format)

    input_data = anthropic_agent._build_input(messages, system_prompt)
    assert input_data["tools"] == claude_format

def test_additional_model_request_fields():
    """Test that additional_model_request_fields are properly added to the model input."""
    # Test with thinking parameter
    thinking_config = {"type": "enabled", "budget_tokens": 2000}
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        additional_model_request_fields={"thinking": thinking_config}
    )

    anthropic_agent = AnthropicAgent(options)
    messages = [{"role": "user", "content": "Test message"}]
    system_prompt = "Test system prompt"

    # Test with thinking
    input_data = anthropic_agent._build_input(messages, system_prompt)
    assert input_data["thinking"] == thinking_config
    
    # Test with multiple additional fields
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        additional_model_request_fields={
            "thinking": thinking_config,
            "custom_param": "custom_value",
            "metadata": {"source": "unit_test"}
        }
    )
    
    anthropic_agent = AnthropicAgent(options)
    input_data = anthropic_agent._build_input(messages, system_prompt)
    
    # Verify all additional fields are present
    assert input_data["thinking"] == thinking_config
    assert input_data["custom_param"] == "custom_value"
    assert input_data["metadata"] == {"source": "unit_test"}
    
    # Verify priority: additional_model_request_fields should override default values
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        additional_model_request_fields={
            "temperature": 0.8  # Override default temperature
        },
        inference_config={"temperature": 0.5}  # This should be overridden
    )
    
    anthropic_agent = AnthropicAgent(options)
    input_data = anthropic_agent._build_input(messages, system_prompt)
    
    # Verify the additional_model_request_fields value takes precedence
    assert input_data["temperature"] == 0.8

def test_get_max_recursions():
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent"
    )

    anthropic_agent = AnthropicAgent(options)

    # Test without tool_config
    assert anthropic_agent._get_max_recursions() == 1

    # Test with tool_config but no toolMaxRecursions
    anthropic_agent.tool_config = {"tool": MagicMock()}
    assert anthropic_agent._get_max_recursions() == 5  # default

    # Test with custom toolMaxRecursions
    anthropic_agent.tool_config = {"tool": MagicMock(), "toolMaxRecursions": 3}
    assert anthropic_agent._get_max_recursions() == 3

@pytest.mark.asyncio
async def test_process_tool_block_with_handler():
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent"
    )

    anthropic_agent = AnthropicAgent(options)

    # Mock llm_response and conversation
    llm_response = MagicMock()
    conversation = [{"role": "user", "content": "Test message"}]

    # Test with useToolHandler
    mock_tool_handler = AsyncMock(return_value={"role": "tool", "content": "Tool response"})
    anthropic_agent.tool_config = {"useToolHandler": mock_tool_handler}

    tool_response = await anthropic_agent._process_tool_block(llm_response, conversation)

    mock_tool_handler.assert_called_once_with(llm_response, conversation)
    assert tool_response == {"role": "tool", "content": "Tool response"}

    # Test with AgentTools
    mock_agent_tools = MagicMock(spec=AgentTools)
    mock_agent_tools.tool_handler = AsyncMock(return_value={"role": "tool", "content": "AgentTools response"})
    anthropic_agent.tool_config = {"tool": mock_agent_tools}

    tool_response = await anthropic_agent._process_tool_block(llm_response, conversation)

    mock_agent_tools.tool_handler.assert_called_once_with(AgentProviderType.ANTHROPIC.value, llm_response, conversation, {'agent_name': 'TestAgent', 'agent_tracking_info': None})
    assert tool_response == {"role": "tool", "content": "AgentTools response"}

    # Test with invalid tool config
    anthropic_agent.tool_config = {"tool": "invalid"}

    with pytest.raises(ValueError, match="You must use AgentTools class when not providing a custom tool handler"):
        await anthropic_agent._process_tool_block(llm_response, conversation)

@pytest.mark.asyncio
async def test_handle_single_response_with_tools():
    # Create a mock Anthropic client
    mock_client = MagicMock()

    # First response with tool_use
    first_response = MagicMock()
    first_response.content = [MagicMock(type="tool_use", text="Using tool")]

    # Second response without tool_use
    second_response = MagicMock()
    second_response.content = [MagicMock(type="text", text="Final response")]

    # Mock handle_single_response to return first_response then second_response
    handle_single_response_mock = AsyncMock(side_effect=[first_response, second_response])

    # Mock _process_tool_block
    process_tool_block_mock = AsyncMock(return_value={"role": "tool", "content": "Tool response"})

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        tool_config={"tool": MagicMock(spec=AgentTools)}
    )

    anthropic_agent = AnthropicAgent(options)
    anthropic_agent.client = mock_client
    anthropic_agent.handle_single_response = handle_single_response_mock
    anthropic_agent._process_tool_block = process_tool_block_mock

    input_data = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [{"role": "user", "content": "Test message"}]
    }

    messages = [{"role": "user", "content": "Test message"}]

    response = await anthropic_agent._handle_single_response_loop(input_data, messages, 3)

    # Check that handle_single_response was called twice
    assert handle_single_response_mock.call_count == 2

    # Check that _process_tool_block was called once
    process_tool_block_mock.assert_called_once()

    # Check that the final response is returned
    if hasattr(response.content[0], 'text'):
        assert response.content[0].text == "Final response"
    else:
        assert response.content[0]["text"] == "Final response"

@pytest.mark.asyncio
async def test_handle_streaming_response():
    """Test the streaming response functionality by directly patching the method."""
    from agent_squad.agents.anthropic_agent import AgentStreamResponse

    # Create the agent with streaming enabled
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        streaming=True
    )

    anthropic_agent = AnthropicAgent(options)

    # Create a simple async generator function that mimics the behavior of handle_streaming_response
    async def mock_streaming_response(input_data):
        mock_content = [MagicMock(type="text", text="Final accumulated response")]
        mock_final_message = MagicMock()
        mock_final_message.content = mock_content
        # First yield: text chunk with no final_message
        yield AgentStreamResponse(text="Streaming chunk 1", final_message=None)
        yield AgentStreamResponse(text="Streaming chunk 2", final_message=None)
        yield AgentStreamResponse(text="", final_message=mock_final_message)

    # Patch the handle_streaming_response method
    with patch.object(anthropic_agent, 'handle_streaming_response', return_value=mock_streaming_response({})):
        # Call process_request which will use our mocked handle_streaming_response
        response_generator = await anthropic_agent.process_request(
            'Test prompt', 'user', 'session', [], {}
        )

        # Collect all responses
        responses = []
        async for response in response_generator:
            responses.append(response)

        # Verify we got the expected pattern of responses
        assert len(responses) == 3
        assert responses[0].text == "Streaming chunk 1"
        assert responses[0].final_message is None

        assert responses[1].text == "Streaming chunk 2"
        assert responses[1].final_message is None

        assert responses[2].text == ""
        assert responses[2].final_message.content[0]["text"] == "Final accumulated response"


@pytest.mark.asyncio
async def test_process_with_strategy():
    """Test strategy selection between streaming and non-streaming responses."""
    from agent_squad.agents.anthropic_agent import AgentStreamResponse

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent"
    )

    anthropic_agent = AnthropicAgent(options)

    # Create a single response for non-streaming path
    single_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Single response"}]
    )

    # Mock the non-streaming handler
    anthropic_agent._handle_single_response_loop = AsyncMock(return_value=single_response)

    # Create a streaming response for the streaming path
    async def stream_generator():
        yield AgentStreamResponse(text="Streaming chunk 1", final_message=None)
        yield AgentStreamResponse(text="Streaming chunk 2", final_message=None)

        final_message = ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Final streaming response"}]
        )
        yield AgentStreamResponse(text="", final_message=final_message)

    # Mock the streaming handler
    anthropic_agent._handle_streaming = AsyncMock(return_value=stream_generator())

    # Setup input and messages
    input_data = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [{"role": "user", "content": "Test message"}]
    }
    messages = [{"role": "user", "content": "Test message"}]

    # Test with streaming=False
    response = await anthropic_agent._process_with_strategy(False, input_data, messages, {"agent_tracking":1234})

    # Verify the non-streaming handler was called
    anthropic_agent._handle_single_response_loop.assert_called_once_with(input_data, messages, 1, {"agent_tracking":1234})
    assert response.content[0]["text"] == "Single response"

    # Reset the mock
    anthropic_agent._handle_single_response_loop.reset_mock()

    # Test with streaming=True
    response_generator = await anthropic_agent._process_with_strategy(True, input_data, messages)

    # Verify the streaming handler was called
    assert anthropic_agent._handle_streaming.call_count == 1

    # Collect responses from the generator
    responses = []
    async for response in response_generator:
        responses.append(response)

    # Verify we got the expected streaming responses
    assert len(responses) == 3
    assert responses[0].text == "Streaming chunk 1"
    assert responses[1].text == "Streaming chunk 2"
    assert responses[2].final_message.content[0]["text"] == "Final streaming response"


@pytest.mark.asyncio
async def test_handle_single_response_error():
    # Create a mock client that raises an exception
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")

    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent"
    )

    anthropic_agent = AnthropicAgent(options)
    anthropic_agent.client = mock_client

    # Test error handling
    with pytest.raises(Exception, match="API error"):
        await anthropic_agent.handle_single_response({"messages": [{'text':'this is the question'}]})


@pytest.mark.asyncio
async def test_handle_streaming_response_implementation():
    """Test the internal implementation of handle_streaming_response."""
    from agent_squad.agents.anthropic_agent import AgentStreamResponse, Logger

    # Create agent with streaming enabled
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        streaming=True
    )

    anthropic_agent = AnthropicAgent(options)
    anthropic_agent.callbacks = MagicMock()

    anthropic_agent.callbacks.on_llm_new_token = AsyncMock()
    anthropic_agent.callbacks.on_llm_start = AsyncMock()
    anthropic_agent.callbacks.on_llm_end = AsyncMock()

    # Create a mock custom stream class that acts as both async iterator and context manager
    class MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __init__(self):
            self.events = [
                type('Event', (), {'type': 'text', 'text': 'Text chunk 1'}),
                type('Event', (), {'type': 'input_json', 'partial_json': '{"key":"value"}'}),
                type('Event', (), {'type': 'content_block_stop'})
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index < len(self.events):
                event = self.events[self.index]
                self.index += 1
                return event
            raise StopAsyncIteration

        async def get_final_message(self):
            message = MagicMock()
            message.content = [{"text": "Final message text"}]
            return message

    # Test successful streaming
    anthropic_agent.client = MagicMock()
    anthropic_agent.client.messages.stream = MagicMock(return_value=MockStream())

    # Call the method
    input_data = {"messages": [{"role": "user", "content": "Test prompt"}]}

    # Collect responses
    responses = []
    async for chunk in anthropic_agent.handle_streaming_response(input_data):
        responses.append(chunk)

    # Verify responses
    assert len(responses) == 2
    assert responses[0].text == "Text chunk 1"
    assert responses[0].final_message is None
    assert responses[1].final_message.content[0]["text"] == "Final message text"

    # Verify callback
    anthropic_agent.callbacks.on_llm_new_token.assert_called_once_with("Text chunk 1")

    # Test error path
    with patch.object(anthropic_agent.client.messages, 'stream', side_effect=Exception("Stream error")), \
         patch.object(Logger, 'error') as mock_logger:

        # Call the method and expect exception
        with pytest.raises(Exception, match="Stream error"):
            async for _ in anthropic_agent.handle_streaming_response(input_data):
                pass

        # Verify logger was called
        mock_logger.assert_called_once()
        assert "Error getting stream from Anthropic model: Stream error" in mock_logger.call_args[0][0]

@pytest.mark.asyncio
async def test_handle_streaming_with_tool_use():
    """Test the streaming response with tool usage."""
    from agent_squad.agents.anthropic_agent import AgentStreamResponse

    # Create agent with streaming enabled
    options = AnthropicAgentOptions(
        api_key='test-api-key',
        name="TestAgent",
        description="A test agent",
        streaming=True,
        tool_config={
            "tool": MagicMock(spec=AgentTools),
            "toolMaxRecursions": 2
        }
    )

    anthropic_agent = AnthropicAgent(options)

    # Mock _process_tool_block to return a tool response
    tool_response = {"role": "tool", "content": "Tool response"}
    anthropic_agent._process_tool_block = AsyncMock(return_value=tool_response)

    # First response generator - contains toolUse
    async def stream_generator_with_tool():
        # Regular chunks
        yield AgentStreamResponse(text="Streaming with tool", final_message=None)

        mock_content = [MagicMock(type="tool_use", text="Final accumulated response")]
        mock_final_message = MagicMock()
        mock_final_message.content = mock_content
        yield AgentStreamResponse(text="", final_message=mock_final_message)

    # Second response generator - no toolUse
    async def stream_generator_final():
        yield AgentStreamResponse(text="Final streaming", final_message=None)

        mock_content = [MagicMock(type="text", text="Final accumulated response")]
        mock_final_message = MagicMock()
        mock_final_message.content = mock_content
        yield AgentStreamResponse(text="", final_message=mock_final_message)

    # Mock handle_streaming_response to return our generators
    anthropic_agent.handle_streaming_response = MagicMock(
        side_effect=[stream_generator_with_tool(), stream_generator_final()]
    )

    # Call _handle_streaming
    input_data = {"messages": [{"role": "user", "content": "Test message"}]}
    messages = [{"role": "user", "content": "Test message"}]

    response_generator = await anthropic_agent._handle_streaming(input_data, messages, 2)

    # Collect all responses
    responses = []
    async for response in response_generator:
        responses.append(response)

    # Verify we got all the expected responses (4 total)
    assert len(responses) == 3
    assert responses[0].text == "Streaming with tool"
    assert responses[1].final_message is None
    assert responses[2].final_message.content[0]["text"] == "Final accumulated response"

    # Verify _process_tool_block was called with the right parameters
    anthropic_agent._process_tool_block.assert_called_once()

    # Verify the messages list was updated with the tool response
    assert input_data["messages"][-1] == tool_response

def test_empty_tool_config_omits_tools(mock_anthropic):
    """An all-hidden/empty tool provider must not attach an empty tools list."""
    options = AnthropicAgentOptions(
        name="TestAgent", description="A test agent", client=Anthropic(api_key="test"),
        tool_config={"tool": AgentTools([])},
    )
    agent = AnthropicAgent(options)
    json_input = agent._build_input([], "system prompt")
    assert "tools" not in json_input


def test_nonempty_tool_config_attaches_tools(mock_anthropic):
    tools = AgentTools([AgentTool(name="t", func=lambda x: "ok")])
    options = AnthropicAgentOptions(
        name="TestAgent", description="A test agent", client=Anthropic(api_key="test"),
        tool_config={"tool": tools},
    )
    agent = AnthropicAgent(options)
    json_input = agent._build_input([], "system prompt")
    assert "tools" in json_input and len(json_input["tools"]) == 1
