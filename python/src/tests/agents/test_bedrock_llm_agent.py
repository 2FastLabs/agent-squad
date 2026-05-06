import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncIterable
from agent_squad.types import ConversationMessage, ParticipantRole, AgentProviderType
from agent_squad.agents import (
    BedrockLLMAgent,
    BedrockLLMAgentOptions,
    AgentStreamResponse)
from agent_squad.utils import Logger, AgentTools, AgentTool
from agent_squad.retrievers import Retriever


logger = Logger()

@pytest.fixture
def mock_boto3_client():
    with patch('boto3.client') as mock_client:
        yield mock_client

@pytest.fixture
def bedrock_llm_agent(mock_boto3_client):
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        model_id="test-model",
        region="us-west-2",
        streaming=False,
        inference_config={
            'maxTokens': 500,
            'temperature': 0.5,
            'topP': 0.8,
            'stopSequences': []
        },
        guardrail_config={
            'guardrailIdentifier': 'myGuardrailIdentifier',
            'guardrailVersion': 'myGuardrailVersion',
            'trace': 'enabled'
        },
        additional_model_request_fields={
            'thinking': {
                'type': 'enabled',
                'budget_tokens': 2000
            }
        }
    )
    agent = BedrockLLMAgent(options)
    yield agent
    mock_boto3_client.reset_mock()


def test_no_region_init(bedrock_llm_agent, mock_boto3_client):
    mock_boto3_client.reset_mock()
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
    )

    _bedrock_llm_agent = BedrockLLMAgent(options)
    assert mock_boto3_client.called
    any_runtime_call = any(args and args[0] == 'bedrock-runtime' for args, kwargs in mock_boto3_client.call_args_list)
    assert any_runtime_call


def test_custom_system_prompt_with_variable(bedrock_llm_agent, mock_boto3_client):
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        }
    )

    _bedrock_llm_agent = BedrockLLMAgent(options)
    assert(_bedrock_llm_agent.system_prompt == 'This is my new prompt with this value')


def test_custom_system_prompt_with_wrong_variable(bedrock_llm_agent, mock_boto3_client):
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variableT': 'value'}
        }
    )

    _bedrock_llm_agent = BedrockLLMAgent(options)
    assert(_bedrock_llm_agent.system_prompt == 'This is my new prompt with this {{variable}}')


@pytest.mark.asyncio
async def test_process_request_single_response(bedrock_llm_agent, mock_boto3_client):
    mock_response = {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [{'text': 'This is a test response'}]
            }
        }
    }
    mock_boto3_client.return_value.converse.return_value = mock_response

    input_text = "Test question"
    user_id = "test_user"
    session_id = "test_session"
    chat_history = []

    result = await bedrock_llm_agent.process_request(input_text, user_id, session_id, chat_history)

    assert isinstance(result, ConversationMessage)
    assert result.role == ParticipantRole.ASSISTANT.value
    assert result.content[0]['text'] == 'This is a test response'

@pytest.mark.asyncio
async def test_agent_tracking_info_propagation(bedrock_llm_agent, mock_boto3_client):
    # Set up mock response
    mock_response = {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [{'text': 'Test response'}]
            }
        },
        'usage': {'inputTokens': 10, 'outputTokens': 20}
    }
    mock_boto3_client.return_value.converse.return_value = mock_response

    # Set up mock callbacks
    bedrock_llm_agent.callbacks = AsyncMock()
    tracking_info = {'trace_id': '123', 'span_id': '456'}
    bedrock_llm_agent.callbacks.on_agent_start.return_value = tracking_info

    # Call the method
    input_text = "Test tracking"
    user_id = "test_user"
    session_id = "test_session"
    chat_history = []

    await bedrock_llm_agent.process_request(input_text, user_id, session_id, chat_history)

    # Verify on_agent_start was called with correct parameters
    bedrock_llm_agent.callbacks.on_agent_start.assert_called_once()
    agent_start_args = bedrock_llm_agent.callbacks.on_agent_start.call_args[1]
    assert agent_start_args['agent_name'] == bedrock_llm_agent.name
    assert agent_start_args['payload_input'] == input_text
    assert agent_start_args['user_id'] == user_id
    assert agent_start_args['session_id'] == session_id

    # Verify on_llm_start was called with tracking info
    bedrock_llm_agent.callbacks.on_llm_start.assert_called_once()
    llm_start_args = bedrock_llm_agent.callbacks.on_llm_start.call_args[1]
    assert llm_start_args['agent_tracking_info'] == tracking_info

    # Verify on_llm_end was called with tracking info
    bedrock_llm_agent.callbacks.on_llm_end.assert_called_once()
    llm_end_args = bedrock_llm_agent.callbacks.on_llm_end.call_args[1]
    assert llm_end_args['agent_tracking_info'] == tracking_info

    # Verify on_agent_end was called with tracking info
    bedrock_llm_agent.callbacks.on_agent_end.assert_called_once()
    agent_end_args = bedrock_llm_agent.callbacks.on_agent_end.call_args[1]
    assert agent_end_args['agent_tracking_info'] == tracking_info

@pytest.mark.asyncio
async def test_agent_tracking_info_streaming(bedrock_llm_agent, mock_boto3_client):
    # Enable streaming
    bedrock_llm_agent.streaming = True

    # Set up mock stream response
    stream_response = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Test "}}},
            {"contentBlockDelta": {"delta": {"text": "response"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}}
        ]
    }
    mock_boto3_client.return_value.converse_stream.return_value = stream_response

    # Set up mock callbacks
    bedrock_llm_agent.callbacks = AsyncMock()
    tracking_info = {'trace_id': '789', 'span_id': '012'}
    bedrock_llm_agent.callbacks.on_agent_start.return_value = tracking_info

    # Call the method
    input_text = "Test streaming tracking"
    user_id = "test_user"
    session_id = "test_session"
    chat_history = []

    result = await bedrock_llm_agent.process_request(input_text, user_id, session_id, chat_history)

    # Collect all chunks to ensure streaming completes
    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    # Verify on_agent_start was called with correct parameters
    bedrock_llm_agent.callbacks.on_agent_start.assert_called_once()
    agent_start_args = bedrock_llm_agent.callbacks.on_agent_start.call_args[1]
    assert agent_start_args['agent_name'] == bedrock_llm_agent.name
    assert agent_start_args['payload_input'] == input_text

    # Verify on_llm_start was called with tracking info
    bedrock_llm_agent.callbacks.on_llm_start.assert_called_once()
    llm_start_args = bedrock_llm_agent.callbacks.on_llm_start.call_args[1]
    assert llm_start_args['agent_tracking_info'] == tracking_info

    # Verify on_llm_new_token was called with tracking info
    assert bedrock_llm_agent.callbacks.on_llm_new_token.call_count == 2
    for call in bedrock_llm_agent.callbacks.on_llm_new_token.call_args_list:
        token_args = call[1]
        assert token_args['agent_tracking_info'] == tracking_info

    # Verify on_llm_end was called with tracking info
    bedrock_llm_agent.callbacks.on_llm_end.assert_called_once()
    llm_end_args = bedrock_llm_agent.callbacks.on_llm_end.call_args[1]
    assert llm_end_args['agent_tracking_info'] == tracking_info

    # Verify on_agent_end was called with tracking info
    bedrock_llm_agent.callbacks.on_agent_end.assert_called_once()
    agent_end_args = bedrock_llm_agent.callbacks.on_agent_end.call_args[1]
    assert agent_end_args['agent_tracking_info'] == tracking_info


@pytest.mark.asyncio
async def test_process_request_streaming(bedrock_llm_agent, mock_boto3_client):
    bedrock_llm_agent.streaming = True
    mock_stream_response = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "This "}}},
            {"contentBlockDelta": {"delta": {"text": "is "}}},
            {"contentBlockDelta": {"delta": {"text": "a "}}},
            {"contentBlockDelta": {"delta": {"text": "test "}}},
            {"contentBlockDelta": {"delta": {"text": "response"}}},
            {"contentBlockStop"}
        ]
    }
    mock_boto3_client.return_value.converse_stream.return_value = mock_stream_response

    input_text = "Test question"
    user_id = "test_user"
    session_id = "test_session"
    chat_history = []

    result = await bedrock_llm_agent.process_request(input_text, user_id, session_id, chat_history)

    assert isinstance(result, AsyncIterable)

    async for chunk in result:
        assert isinstance(chunk, AgentStreamResponse)
        if chunk.final_message:
            assert chunk.final_message.role == ParticipantRole.ASSISTANT.value
            assert chunk.final_message.content[0]['text'] == 'This is a test response'


@pytest.mark.asyncio
async def test_process_request_with_tool_use(bedrock_llm_agent, mock_boto3_client):
    async def _handler(message, conversation):
        return ConversationMessage(role=ParticipantRole.ASSISTANT, content=[{'text': 'Tool response'}])
    bedrock_llm_agent.tool_config = {
        "tool": [
            AgentTool(name='test_tool', func=_handler, description='This is a test handler')
        ],
        "toolMaxRecursions": 2,
        "useToolHandler": AsyncMock()
    }

    mock_responses = [
        {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [{'toolUse': {'name': 'test_tool'}}]
                }
            }
        },
        {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [{'text': 'Final response'}]
                }
            }
        }
    ]
    mock_boto3_client.return_value.converse.side_effect = mock_responses

    input_text = "Test question"
    user_id = "test_user"
    session_id = "test_session"
    chat_history = []

    result = await bedrock_llm_agent.process_request(input_text, user_id, session_id, chat_history)

    assert isinstance(result, ConversationMessage)
    assert result.role == ParticipantRole.ASSISTANT.value
    assert result.content[0]['text'] == 'Final response'
    assert bedrock_llm_agent.tool_config['useToolHandler'].call_count == 1

def test_set_system_prompt(bedrock_llm_agent):
    new_template = "You are a {{role}}. Your task is {{task}}."
    variables = {"role": "test agent", "task": "to run tests"}

    bedrock_llm_agent.set_system_prompt(new_template, variables)

    assert bedrock_llm_agent.prompt_template == new_template
    assert bedrock_llm_agent.custom_variables == variables
    assert bedrock_llm_agent.system_prompt == "You are a test agent. Your task is to run tests."

def test_streaming(mock_boto3_client):
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        },
        streaming=True
    )

    agent = BedrockLLMAgent(options)
    assert(agent.is_streaming_enabled() == True)

    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        },
        streaming=False
    )

    agent = BedrockLLMAgent(options)
    assert(agent.is_streaming_enabled() == False)

    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        custom_system_prompt={
            'template': """This is my new prompt with this {{variable}}""",
            'variables': {'variable': 'value'}
        }
    )

    agent = BedrockLLMAgent(options)
    assert(agent.is_streaming_enabled() == False)


@pytest.mark.asyncio
async def test_prepare_system_prompt_with_retriever(bedrock_llm_agent):
    # Create a mock retriever
    mock_retriever = AsyncMock(spec=Retriever)
    mock_retriever.retrieve_and_combine_results.return_value = "Retrieved context"

    # Update the agent with the retriever
    bedrock_llm_agent.retriever = mock_retriever

    # Call the method
    system_prompt = await bedrock_llm_agent._prepare_system_prompt("Test input")

    # Verify the result and the retriever call
    assert "Retrieved context" in system_prompt
    mock_retriever.retrieve_and_combine_results.assert_called_once_with("Test input")

def test_prepare_tool_config_with_agent_tools(bedrock_llm_agent):
    # Create mock AgentTools
    mock_agent_tools = Mock(spec=AgentTools)
    mock_agent_tools.to_bedrock_format.return_value = [{"name": "test_tool"}]

    # Set up the tool_config
    bedrock_llm_agent.tool_config = {"tool": mock_agent_tools}

    # Call the method
    result = bedrock_llm_agent._prepare_tool_config()

    # Verify the result
    assert result == {"tools": [{"name": "test_tool"}]}
    mock_agent_tools.to_bedrock_format.assert_called_once()

def test_prepare_tool_config_with_agent_tool_list(bedrock_llm_agent):
    # Create mock AgentTool
    mock_agent_tool = Mock(spec=AgentTool)
    mock_agent_tool.to_bedrock_format.return_value = {"name": "test_tool"}

    # Also include a non-AgentTool item
    direct_tool_dict = {"name": "direct_tool"}

    # Set up the tool_config
    bedrock_llm_agent.tool_config = {"tool": [mock_agent_tool, direct_tool_dict]}

    # Call the method
    result = bedrock_llm_agent._prepare_tool_config()

    # Verify the result
    assert result == {"tools": [{"name": "test_tool"}, {"name": "direct_tool"}]}
    mock_agent_tool.to_bedrock_format.assert_called_once()

def test_prepare_tool_config_with_invalid_config(bedrock_llm_agent):
    # Set up an invalid tool_config
    bedrock_llm_agent.tool_config = {"tool": "invalid"}

    # Call the method and check for exception
    with pytest.raises(RuntimeError, match="Invalid tool config"):
        bedrock_llm_agent._prepare_tool_config()


@pytest.mark.asyncio
async def test_handle_single_response_error(bedrock_llm_agent, mock_boto3_client):
    # Set up the mock to raise an exception
    mock_boto3_client.return_value.converse.side_effect = Exception("Test error")

    # Call the method and check for exception
    with pytest.raises(Exception, match="Test error"):
        await bedrock_llm_agent.handle_single_response({'messages':[{'text'}]}, {})

@pytest.mark.asyncio
async def test_handle_streaming_response_error(bedrock_llm_agent, mock_boto3_client):
    # Set up the mock to raise an exception
    mock_boto3_client.return_value.converse_stream.side_effect = Exception("Test error")

    # Call the method and check for exception
    with pytest.raises(Exception, match="Test error"):
        async for _ in bedrock_llm_agent.handle_streaming_response({'messages':[{'text'}]},  {}):
            pass


@pytest.mark.asyncio
async def test_process_tool_block_with_agent_tools(bedrock_llm_agent):
    # Create a mock AgentTools
    mock_agent_tools = AsyncMock(spec=AgentTools)
    expected_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Tool response"}]
    )
    mock_agent_tools.tool_handler.return_value = expected_response

    # Set up the tool_config
    bedrock_llm_agent.tool_config = {"tool": mock_agent_tools}

    # Create a test LLM response with toolUse
    llm_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"toolUse": {"name": "test_tool", "input": {"param": "value"}}}]
    )

    # Call the method
    result = await bedrock_llm_agent._process_tool_block(llm_response, [], {"agent_start_id":1234})

    # Verify the result
    assert result == expected_response
    mock_agent_tools.tool_handler.assert_called_once_with(
        AgentProviderType.BEDROCK.value, llm_response, [], {'agent_name':'TestAgent', 'agent_tracking_info': {"agent_start_id":1234}}
    )

@pytest.mark.asyncio
async def test_process_tool_block_with_invalid_tool(bedrock_llm_agent):
    # Set up an invalid tool configuration
    bedrock_llm_agent.tool_config = {"tool": "invalid"}

    # Create a test LLM response with toolUse
    llm_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"toolUse": {"name": "test_tool", "input": {"param": "value"}}}]
    )

    # Call the method and check for exception
    with pytest.raises(ValueError, match="You must use AgentTools class"):
        await bedrock_llm_agent._process_tool_block(llm_response, [])


@pytest.mark.asyncio
async def test_handle_streaming_with_tool_use(bedrock_llm_agent, mock_boto3_client):
    # Enable streaming
    bedrock_llm_agent.streaming = True

    # Set up the tool handler
    async def mock_tool_handler(message, conversation):
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Tool response"}]
        )

    bedrock_llm_agent.tool_config = {
        "tool": AgentTools(tools=[]),
        "useToolHandler": mock_tool_handler
    }

    # First response with tool use
    stream_response1 = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test_tool"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "{\"param\":"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "\"value\"}"}}}},
            {"contentBlockStop": {}}
        ]
    }

    # Second response after tool use
    stream_response2 = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Final"}}},
            {"contentBlockDelta": {"delta": {"text": " response"}}},
            {"contentBlockStop": {}}
        ]
    }

    mock_boto3_client.return_value.converse_stream.side_effect = [stream_response1, stream_response2]

    # Call the method
    input_text = "Test with tool"
    user_id = "test_user"
    session_id = "test_session"
    chat_history = []

    result = await bedrock_llm_agent.process_request(input_text, user_id, session_id, chat_history)

    # Verify it's an AsyncIterable
    assert isinstance(result, AsyncIterable)

    # Collect all chunks
    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    # Verify we get the expected number of chunks
    assert len(chunks) > 0

    # Verify the final message in the last chunk
    final_chunks = [chunk for chunk in chunks if chunk.final_message is not None]
    assert len(final_chunks) > 0

    # Verify converse_stream was called twice (first for tool use, then for final response)
    assert mock_boto3_client.return_value.converse_stream.call_count == 2

@pytest.mark.asyncio
async def test_handle_single_response_no_output(bedrock_llm_agent, mock_boto3_client):
    # Set up mock to return response with no output
    mock_boto3_client.return_value.converse.return_value = {"not_output": {}}

    # Call the method and check for exception
    with pytest.raises(ValueError, match="No output received from Bedrock model"):
        await bedrock_llm_agent.handle_single_response({'messages':[{'role':'user','content':'text'}]}, {})

@pytest.mark.asyncio
async def test_handle_streaming_with_text_response(bedrock_llm_agent, mock_boto3_client):
    # Set up stream response with only text content
    stream_response = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "This "}}},
            {"contentBlockDelta": {"delta": {"text": "is "}}},
            {"contentBlockDelta": {"delta": {"text": "a "}}},
            {"contentBlockStop": {}}
        ]
    }

    mock_boto3_client.return_value.converse_stream.return_value = stream_response

    # Initialize callbacks
    bedrock_llm_agent.callbacks = AsyncMock()

    converse_input = {
        'system':[{'text':'this is the system messages'}],
        'messages':[{'text'}]
    }

    # Call the method
    chunks = []
    response = bedrock_llm_agent.handle_streaming_response(converse_input, {})

    async for chunk in response:
        chunks.append(chunk)

    # Verify we got the expected chunks
    assert len(chunks) == 4  # 3 text chunks + final message

    # Verify callbacks were called for each text chunk
    assert bedrock_llm_agent.callbacks.on_llm_new_token.call_count == 3

    # Verify the last chunk has the final message
    assert chunks[-1].final_message is not None
    assert chunks[-1].final_message.role == ParticipantRole.ASSISTANT.value
    assert chunks[-1].final_message.content[0]["text"] == "This is a "

@pytest.mark.asyncio
async def test_handle_streaming_response_no_output(bedrock_llm_agent, mock_boto3_client):
    # Set up the mock to return an empty stream response
    mock_boto3_client.return_value.converse_stream.return_value = {"stream": []}

    # Initialize callbacks
    bedrock_llm_agent.callbacks = AsyncMock()

    converse_input = {
        'system':[{'text':'this is the system messages'}],
        'messages':[{'text': 'user message'}]
    }

    # Call the method and collect chunks
    chunks = []
    response = bedrock_llm_agent.handle_streaming_response(converse_input, {})

    async for chunk in response:
        chunks.append(chunk)

    # Verify we got a final message with empty content
    assert len(chunks) == 1  # Only the final message
    assert chunks[0].final_message is not None
    assert chunks[0].final_message.role == ParticipantRole.ASSISTANT.value
    assert len(chunks[0].final_message.content) == 0

@pytest.mark.asyncio
async def test_handle_streaming_with_metadata(bedrock_llm_agent, mock_boto3_client):
    # Set up stream response with metadata
    stream_response = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Response text"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}
        ]
    }

    mock_boto3_client.return_value.converse_stream.return_value = stream_response

    # Initialize callbacks
    bedrock_llm_agent.callbacks = AsyncMock()

    converse_input = {
        'system':[{'text':'system message'}],
        'messages':[{'text': 'user message'}]
    }

    # Call the method
    chunks = []
    response = bedrock_llm_agent.handle_streaming_response(converse_input, {})

    async for chunk in response:
        chunks.append(chunk)

    # Verify the callbacks were called with the metadata
    assert bedrock_llm_agent.callbacks.on_llm_end.call_count == 1
    call_args = bedrock_llm_agent.callbacks.on_llm_end.call_args[1]
    assert 'usage' in call_args
    assert call_args['usage'] == {"inputTokens": 10, "outputTokens": 5}

@pytest.mark.asyncio
async def test_get_max_recursions(bedrock_llm_agent):
    # Test without tool config
    bedrock_llm_agent.tool_config = None
    assert bedrock_llm_agent._get_max_recursions() == 1

    # Test with tool config but without toolMaxRecursions
    bedrock_llm_agent.tool_config = {"tool": Mock()}
    assert bedrock_llm_agent._get_max_recursions() == bedrock_llm_agent.default_max_recursions

    # Test with tool config and custom toolMaxRecursions
    bedrock_llm_agent.tool_config = {"tool": Mock(), "toolMaxRecursions": 5}
    assert bedrock_llm_agent._get_max_recursions() == 5


def test_update_system_prompt(bedrock_llm_agent):
    # Set initial template and variables
    bedrock_llm_agent.prompt_template = "Hello {{name}}, welcome to {{service}}!"
    bedrock_llm_agent.custom_variables = {"name": "User", "service": "Testing"}

    # Call the method
    bedrock_llm_agent.update_system_prompt()

    # Verify the result
    assert bedrock_llm_agent.system_prompt == "Hello User, welcome to Testing!"

    # Test with list variable
    bedrock_llm_agent.custom_variables = {"name": "User", "service": ["Testing", "Service"]}
    bedrock_llm_agent.update_system_prompt()
    assert bedrock_llm_agent.system_prompt == "Hello User, welcome to Testing\nService!"

def test_prepare_conversation(bedrock_llm_agent):
    # Create test data
    input_text = "Hello, how are you?"
    chat_history = [
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": "Previous message"}]
        ),
        ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Previous response"}]
        )
    ]

    # Call the method
    result = bedrock_llm_agent._prepare_conversation(input_text, chat_history)

    # Verify the result
    assert len(result) == 3
    assert result[0].role == ParticipantRole.USER.value
    assert result[0].content[0]["text"] == "Previous message"
    assert result[1].role == ParticipantRole.ASSISTANT.value
    assert result[1].content[0]["text"] == "Previous response"
    assert result[2].role == ParticipantRole.USER.value
    assert result[2].content[0]["text"] == "Hello, how are you?"

def test_build_conversation_command(bedrock_llm_agent):
    # Set up test data
    conversation = [
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": "Test message"}]
        )
    ]
    system_prompt = "Test system prompt"

    # Set up tool config for testing
    mock_agent_tools = Mock(spec=AgentTools)
    mock_agent_tools.to_bedrock_format.return_value = [{"name": "test_tool"}]
    bedrock_llm_agent.tool_config = {"tool": mock_agent_tools}

    # Call the method
    result = bedrock_llm_agent._build_conversation_command(conversation, system_prompt)

    # Verify the result
    assert result["modelId"] == bedrock_llm_agent.model_id
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"][0]["text"] == "Test message"
    assert result["system"][0]["text"] == "Test system prompt"
    assert "inferenceConfig" in result
    assert result["inferenceConfig"]["maxTokens"] == bedrock_llm_agent.inference_config["maxTokens"]
    assert result["inferenceConfig"]["temperature"] == bedrock_llm_agent.inference_config["temperature"]
    
    # Check for topP only if it exists in the inference_config
    # (it's removed when reasoning_config with thinking is enabled)
    if "topP" in bedrock_llm_agent.inference_config:
        assert result["inferenceConfig"]["topP"] == bedrock_llm_agent.inference_config["topP"]
    else:
        assert "topP" not in result["inferenceConfig"]
        
    assert "guardrailConfig" in result
    assert "toolConfig" in result
    assert result["toolConfig"]["tools"] == [{"name": "test_tool"}]
    assert "additionalModelRequestFields" in result
    assert "thinking" in result["additionalModelRequestFields"]

    # Test without tool config
    bedrock_llm_agent.tool_config = None
    result = bedrock_llm_agent._build_conversation_command(conversation, system_prompt)
    assert "toolConfig" not in result

@pytest.fixture
def client_fixture():
    # Create a mock client
    mock_client = Mock()
    return mock_client

def test_client_provided(client_fixture):
    # Test initialization with provided client
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        client=client_fixture
    )

    agent = BedrockLLMAgent(options)
    assert agent.client is client_fixture


def test_additional_model_request_fields(mock_boto3_client):
    """Test that additional_model_request_fields are properly added to the model input."""
    # Test with thinking parameter
    thinking_config = {"type": "enabled", "budget_tokens": 2000}
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        model_id="test-model",
        additional_model_request_fields={"thinking": thinking_config}
    )

    agent = BedrockLLMAgent(options)
    conversation = [ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"text": "Test message"}]
    )]
    system_prompt = "Test system prompt"

    # Test with thinking
    result = agent._build_conversation_command(conversation, system_prompt)
    assert result["additionalModelRequestFields"]["thinking"] == thinking_config
    
    # Verify topP is removed when thinking is enabled
    assert "topP" not in result["inferenceConfig"]
    
    # Test with multiple additional fields
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        model_id="test-model",
        additional_model_request_fields={
            "thinking": thinking_config,
            "custom_param": "custom_value",
            "metadata": {"source": "unit_test"}
        }
    )
    
    agent = BedrockLLMAgent(options)
    result = agent._build_conversation_command(conversation, system_prompt)
    
    # Verify all additional fields are present
    assert result["additionalModelRequestFields"]["thinking"] == thinking_config
    assert result["additionalModelRequestFields"]["custom_param"] == "custom_value"
    assert result["additionalModelRequestFields"]["metadata"] == {"source": "unit_test"}
    
    # Test without thinking - topP should be present
    options = BedrockLLMAgentOptions(
        name="TestAgent",
        description="A test agent",
        model_id="test-model",
        additional_model_request_fields={
            "custom_param": "custom_value"
        }
    )
    
    agent = BedrockLLMAgent(options)
    result = agent._build_conversation_command(conversation, system_prompt)
    
    # Verify topP is present when thinking is not enabled
    assert "topP" in result["inferenceConfig"]
    assert result["inferenceConfig"]["topP"] == 0.9  # Default value


@pytest.mark.asyncio
async def test_tool_conversation_collected_during_tool_use(bedrock_llm_agent, mock_boto3_client):
    """Test that intermediate tool messages are collected in tool_conversation."""
    tool_use_response = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}]}}]
    )

    bedrock_llm_agent.tool_config = {
        "tool": [
            AgentTool(name='test_tool', func=lambda: None, description='Test tool')
        ],
        "toolMaxRecursions": 5,
        "useToolHandler": AsyncMock(return_value=tool_use_response)
    }

    mock_responses = [
        {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'text': 'Let me check'},
                        {'toolUse': {'toolUseId': '123', 'name': 'test_tool', 'input': {}}}
                    ]
                }
            }
        },
        {
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [{'text': 'Here is the result'}]
                }
            }
        }
    ]
    mock_boto3_client.return_value.converse.side_effect = mock_responses

    result = await bedrock_llm_agent.process_request(
        "Test question", "test_user", "test_session", []
    )

    # Verify the final response is clean (no toolUse)
    assert isinstance(result, ConversationMessage)
    assert result.content[0]['text'] == 'Here is the result'
    assert not any("toolUse" in c for c in result.content)

    # Verify tool_conversation collected intermediate messages
    assert len(bedrock_llm_agent.tool_conversation) == 2
    # First: assistant message with toolUse
    assert any("toolUse" in c for c in bedrock_llm_agent.tool_conversation[0].content)
    # Second: user message with toolResult
    assert any("toolResult" in c for c in bedrock_llm_agent.tool_conversation[1].content)


@pytest.mark.asyncio
async def test_max_recursions_exhaustion_returns_clean_response(bedrock_llm_agent, mock_boto3_client):
    """Test that when max_recursions is exhausted, the response has no toolUse blocks."""
    tool_use_response = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}]}}]
    )

    bedrock_llm_agent.tool_config = {
        "tool": [
            AgentTool(name='test_tool', func=lambda: None, description='Test tool')
        ],
        "toolMaxRecursions": 1,  # Only 1 recursion allowed
        "useToolHandler": AsyncMock(return_value=tool_use_response)
    }

    # Model always returns toolUse - will exhaust max_recursions
    mock_boto3_client.return_value.converse.return_value = {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [
                    {'text': 'Let me check'},
                    {'toolUse': {'toolUseId': '123', 'name': 'test_tool', 'input': {}}}
                ]
            }
        }
    }

    result = await bedrock_llm_agent.process_request(
        "Test question", "test_user", "test_session", []
    )

    # Verify the response is clean - no toolUse blocks
    assert isinstance(result, ConversationMessage)
    assert not any("toolUse" in c for c in result.content)
    # Should have text content (either original text or fallback message)
    assert any("text" in c for c in result.content)


@pytest.mark.asyncio
async def test_tool_conversation_empty_when_no_tools(bedrock_llm_agent, mock_boto3_client):
    """Test that tool_conversation is empty when no tools are used."""
    mock_boto3_client.return_value.converse.return_value = {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [{'text': 'Simple response'}]
            }
        }
    }

    result = await bedrock_llm_agent.process_request(
        "Test question", "test_user", "test_session", []
    )

    assert isinstance(result, ConversationMessage)
    assert result.content[0]['text'] == 'Simple response'
    assert len(bedrock_llm_agent.tool_conversation) == 0


@pytest.mark.asyncio
async def test_streaming_tool_conversation_collected(bedrock_llm_agent, mock_boto3_client):
    """Test that tool_conversation is collected during streaming with tool use."""
    bedrock_llm_agent.streaming = True

    async def mock_tool_handler(message, conversation):
        return ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}]}}]
        )

    bedrock_llm_agent.tool_config = {
        "tool": AgentTools(tools=[]),
        "useToolHandler": mock_tool_handler
    }

    # First response with tool use
    stream_response1 = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test_tool"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "{\"param\":"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "\"value\"}"}}}},
            {"contentBlockStop": {}}
        ]
    }

    # Second response after tool use (text only)
    stream_response2 = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Final response"}}},
            {"contentBlockStop": {}}
        ]
    }

    mock_boto3_client.return_value.converse_stream.side_effect = [stream_response1, stream_response2]

    result = await bedrock_llm_agent.process_request(
        "Test with tool", "test_user", "test_session", []
    )

    # Consume the stream
    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    # Verify tool_conversation was populated during streaming
    assert len(bedrock_llm_agent.tool_conversation) == 2
    # First: assistant message with toolUse
    assert any("toolUse" in c for c in bedrock_llm_agent.tool_conversation[0].content)
    # Second: user message with toolResult
    assert any("toolResult" in c for c in bedrock_llm_agent.tool_conversation[1].content)
