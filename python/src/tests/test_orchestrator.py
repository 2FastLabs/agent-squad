import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncIterable
import pytest_asyncio
from dataclasses import dataclass

from agent_squad.types import (
    ConversationMessage,
    ParticipantRole,
    AgentSquadConfig,
    TimestampedMessage
)
from agent_squad.classifiers import Classifier, ClassifierResult
from agent_squad.agents import (
    Agent,
    AgentStreamResponse,
    AgentResponse,
    AgentProcessingResult
)
from agent_squad.storage import ChatStorage, InMemoryChatStorage
from agent_squad.utils.logger import Logger
from agent_squad.orchestrator import AgentSquad

@pytest.fixture
def mock_boto3_client():
    with patch('boto3.client') as mock_client:
        yield mock_client

# Fixtures
@pytest.fixture
def mock_logger():
    return Mock(spec=Logger)

@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=ChatStorage)
    storage.fetch_chat = AsyncMock(return_value=[])
    storage.fetch_all_chats = AsyncMock(return_value=[])
    storage.save_chat_message = AsyncMock()
    return storage

@pytest.fixture
def mock_classifier():
    classifier = AsyncMock(spec=Classifier)
    classifier.set_agents = Mock()
    return classifier

@pytest.fixture
def mock_agent():
    agent = AsyncMock(spec=Agent)
    agent.id = "test_agent"
    agent.name = "Test Agent"
    agent.description = "Test Agent Description"
    agent.save_chat = True
    agent.is_streaming_enabled = Mock(return_value=False)
    return agent

@pytest.fixture
def mock_streaming_agent():
    agent = AsyncMock(spec=Agent)
    agent.id = "streaming_agent"
    agent.name = "Streaming Agent"
    agent.description = "Streaming Agent Description"
    agent.save_chat = True
    agent.is_streaming_enabled = Mock(return_value=True)
    return agent

@pytest.fixture
def orchestrator(mock_storage, mock_classifier, mock_logger, mock_agent, mock_boto3_client):
    return AgentSquad(
        storage=mock_storage,
        classifier=mock_classifier,
        logger=mock_logger,
        default_agent=mock_agent
    )

def test_init_with_dict_options(mock_boto3_client):
    options = {"MAX_MESSAGE_PAIRS_PER_AGENT": 10}
    orchestrator = AgentSquad(
        options=options,
        classifier=Mock(spec=Classifier)
    )
    assert orchestrator.config.MAX_MESSAGE_PAIRS_PER_AGENT == 10

def test_init_with_invalid_options(mock_boto3_client):
    with pytest.raises(ValueError):
        AgentSquad(options="invalid")

# Test agent management
def test_add_agent(orchestrator, mock_agent):
    orchestrator.add_agent(mock_agent)
    assert orchestrator.agents[mock_agent.id] == mock_agent
    orchestrator.classifier.set_agents.assert_called_once_with(orchestrator.agents)

def test_add_duplicate_agent(orchestrator, mock_agent):
    orchestrator.add_agent(mock_agent)
    with pytest.raises(ValueError):
        orchestrator.add_agent(mock_agent)

def test_get_all_agents(orchestrator, mock_agent):
    orchestrator.add_agent(mock_agent)
    agents = orchestrator.get_all_agents()
    assert agents[mock_agent.id]["name"] == mock_agent.name
    assert agents[mock_agent.id]["description"] == mock_agent.description

# Test default agent management
def test_get_default_agent(orchestrator, mock_agent):
    assert orchestrator.get_default_agent() == mock_agent

def test_set_default_agent(orchestrator, mock_agent):
    new_agent = AsyncMock(spec=Agent)
    orchestrator.set_default_agent(new_agent)
    assert orchestrator.get_default_agent() == new_agent

# Test request classification
@pytest.mark.asyncio
async def test_classify_request_success(orchestrator, mock_agent):
    expected_result = ClassifierResult(selected_agent=mock_agent, confidence=0.9)
    orchestrator.classifier.classify.return_value = expected_result

    result = await orchestrator.classify_request("test input", "user1", "session1")
    assert result == expected_result

@pytest.mark.asyncio
async def test_classify_request_no_agent_with_default(orchestrator):
    orchestrator.classifier.classify.return_value = ClassifierResult(selected_agent=None, confidence=0)
    orchestrator.config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED = True

    result = await orchestrator.classify_request("test input", "user1", "session1")
    assert result.selected_agent == orchestrator.default_agent

@pytest.mark.asyncio
async def test_classify_request_error(orchestrator):
    orchestrator.classifier.classify.side_effect = Exception("Classification error")

    with pytest.raises(Exception):
        await orchestrator.classify_request("test input", "user1", "session1")

# Test dispatch to agent
@pytest.mark.asyncio
async def test_dispatch_to_agent_success(orchestrator, mock_agent):
    classifier_result = ClassifierResult(selected_agent=mock_agent, confidence=0.9)
    expected_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Test response"}]
    )
    mock_agent.process_request.return_value = expected_response

    response, agent = await orchestrator.dispatch_to_agent({
        "user_input": "test",
        "user_id": "user1",
        "session_id": "session1",
        "classifier_result": classifier_result,
        "additional_params": {}
    })

    assert response == expected_response
    assert agent == mock_agent

@pytest.mark.asyncio
async def test_dispatch_to_agent_no_agent(orchestrator):
    classifier_result = ClassifierResult(selected_agent=None, confidence=0)

    response, agent = await orchestrator.dispatch_to_agent({
        "user_input": "test",
        "user_id": "user1",
        "session_id": "session1",
        "classifier_result": classifier_result,
        "additional_params": {}
    })

    assert isinstance(response, ConversationMessage)
    assert "more information" in response.content[0].get('text')
    assert agent is None

# Test streaming functionality
@pytest.mark.asyncio
async def test_agent_process_request_streaming(orchestrator, mock_streaming_agent):
    classifier_result = ClassifierResult(selected_agent=mock_streaming_agent, confidence=0.9)

    async def mock_stream():
        yield AgentStreamResponse(
            chunk="Test chunk",
            final_message=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": "Final message"}]
            )
        )

    mock_streaming_agent.process_request.return_value = mock_stream()

    response = await orchestrator.agent_process_request(
        "test input",
        "user1",
        "session1",
        classifier_result,
        stream_response=True
    )

    assert response.streaming == True
    assert isinstance(response.output, AsyncIterable)

# Test route request
@pytest.mark.asyncio
async def test_route_request_success(orchestrator, mock_agent):
    classifier_result = ClassifierResult(selected_agent=mock_agent, confidence=0.9)
    orchestrator.classifier.classify.return_value = classifier_result

    expected_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Test response"}]
    )
    mock_agent.process_request.return_value = expected_response

    response = await orchestrator.route_request(
        "test input",
        "user1",
        "session1"
    )

    assert response.output == expected_response
    assert response.metadata.agent_id == mock_agent.id

@pytest.mark.asyncio
async def test_route_request_error(orchestrator):
    orchestrator.classifier.classify.side_effect = Exception("Test error")

    response = await orchestrator.route_request(
        "test input",
        "user1",
        "session1"
    )

    assert isinstance(response.output, ConversationMessage)
    assert "Test error" in response.output.content[0].get('text')

# Test chat storage
@pytest.mark.asyncio
async def test_save_message(orchestrator, mock_agent):
    message = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Test message"}]
    )

    await orchestrator.save_message(
        message,
        "user1",
        "session1",
        mock_agent
    )

    orchestrator.storage.save_chat_message.assert_called_once_with(
        "user1",
        "session1",
        mock_agent.id,
        message,
        orchestrator.config.MAX_MESSAGE_PAIRS_PER_AGENT
    )

@pytest.mark.asyncio
async def test_save_messages(orchestrator, mock_agent):
    messages = [
        ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Message 1"}]
        ),
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": "Message 2"}]
        )
    ]

    await orchestrator.save_messages(
        messages,
        "user1",
        "session1",
        mock_agent
    )

    assert orchestrator.storage.save_chat_message.call_count == 2

# Test execution time measurement
@pytest.mark.asyncio
async def test_measure_execution_time(orchestrator):
    async def test_fn():
        return "test result"

    orchestrator.config.LOG_EXECUTION_TIMES = True
    result = await orchestrator.measure_execution_time("test_timer", test_fn)

    assert result == "test result"
    assert "test_timer" in orchestrator.execution_times
    assert isinstance(orchestrator.execution_times["test_timer"], float)

@pytest.mark.asyncio
async def test_measure_execution_time_error(orchestrator):
    async def test_fn():
        raise Exception("Test error")

    orchestrator.config.LOG_EXECUTION_TIMES = True

    with pytest.raises(Exception):
        await orchestrator.measure_execution_time("test_timer", test_fn)

    assert "test_timer" in orchestrator.execution_times
    assert isinstance(orchestrator.execution_times["test_timer"], float)

# Test metadata creation
def test_create_metadata(orchestrator, mock_agent):
    classifier_result = ClassifierResult(selected_agent=mock_agent, confidence=0.9)

    metadata = orchestrator.create_metadata(
        classifier_result,
        "test input",
        "user1",
        "session1",
        {"param1": "value1"}
    )

    assert metadata.user_input == "test input"
    assert metadata.agent_id == mock_agent.id
    assert metadata.agent_name == mock_agent.name
    assert metadata.user_id == "user1"
    assert metadata.session_id == "session1"
    assert metadata.additional_params == {"param1": "value1"}

def test_create_metadata_no_agent(orchestrator):
    metadata = orchestrator.create_metadata(
        None,
        "test input",
        "user1",
        "session1",
        {}
    )

    assert metadata.agent_id == "no_agent_selected"
    assert metadata.agent_name == "No Agent"
    assert "error_type" in metadata.additional_params
    assert metadata.additional_params["error_type"] == "classification_failed"

# Test intermediate tool messages are saved
@pytest.mark.asyncio
async def test_agent_process_request_saves_tool_conversation(orchestrator, mock_agent):
    """Test that intermediate tool messages (toolUse/toolResult) are saved to storage."""
    classifier_result = ClassifierResult(selected_agent=mock_agent, confidence=0.9)

    # Set up tool_conversation on the mock agent
    tool_use_msg = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}]
    )
    tool_result_msg = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}]}}]
    )
    mock_agent.tool_conversation = [tool_use_msg, tool_result_msg]

    final_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Final answer based on tool output"}]
    )
    mock_agent.process_request.return_value = final_response

    response = await orchestrator.agent_process_request(
        "test input", "user1", "session1", classifier_result
    )

    # Verify storage was called for:
    # 1. User message
    # 2. Tool use message (from tool_conversation)
    # 3. Tool result message (from tool_conversation)
    # 4. Final assistant response
    assert orchestrator.storage.save_chat_message.call_count == 4

    # Check the saved messages in order
    calls = orchestrator.storage.save_chat_message.call_args_list
    # First: user message
    assert calls[0][0][3].role == ParticipantRole.USER.value
    assert calls[0][0][3].content[0]['text'] == 'test input'
    # Second: assistant toolUse
    assert calls[1][0][3].role == ParticipantRole.ASSISTANT.value
    assert "toolUse" in calls[1][0][3].content[0]
    # Third: user toolResult
    assert calls[2][0][3].role == ParticipantRole.USER.value
    assert "toolResult" in calls[2][0][3].content[0]
    # Fourth: final assistant response
    assert calls[3][0][3].role == ParticipantRole.ASSISTANT.value
    assert calls[3][0][3].content[0]['text'] == 'Final answer based on tool output'


@pytest.mark.asyncio
async def test_agent_process_request_no_tool_conversation(orchestrator, mock_agent):
    """Test that when no tools are used, only user message and response are saved."""
    classifier_result = ClassifierResult(selected_agent=mock_agent, confidence=0.9)

    # No tool_conversation attribute
    if hasattr(mock_agent, 'tool_conversation'):
        delattr(mock_agent, 'tool_conversation')

    final_response = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "Simple response"}]
    )
    mock_agent.process_request.return_value = final_response

    response = await orchestrator.agent_process_request(
        "test input", "user1", "session1", classifier_result
    )

    # Only user message + final response = 2 saves
    assert orchestrator.storage.save_chat_message.call_count == 2


# Test fallback functionality
def test_get_fallback_result(orchestrator, mock_agent):
    result = orchestrator.get_fallback_result()
    assert result.selected_agent == mock_agent
    assert result.confidence == 0