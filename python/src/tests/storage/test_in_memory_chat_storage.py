import pytest
from unittest.mock import patch
from agent_squad.types import ConversationMessage, TimestampedMessage
from agent_squad.storage import InMemoryChatStorage
from agent_squad.utils import Logger

tmp_logger = Logger()

@pytest.fixture
def mock_logger():
    with patch('agent_squad.utils.logger') as mock:
        yield mock


@pytest.fixture
def storage(mock_logger):
    return InMemoryChatStorage()

@pytest.mark.asyncio
async def test_save_chat_message(storage):
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    message = ConversationMessage(role="user", content="Hello")

    result = await storage.save_chat_message(user_id, session_id, agent_id, message)

    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello"

@pytest.mark.asyncio
async def test_save_consecutive_message(storage):
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    message1 = ConversationMessage(role="user", content="Hello")
    message2 = ConversationMessage(role="user", content="World")

    await storage.save_chat_message(user_id, session_id, agent_id, message1)
    result = await storage.save_chat_message(user_id, session_id, agent_id, message2)

    assert len(result) == 1
    assert result[0].content == "Hello"

@pytest.mark.asyncio
async def test_fetch_chat(storage):
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    message = ConversationMessage(role="user", content="Hello")

    await storage.save_chat_message(user_id, session_id, agent_id, message)
    result = await storage.fetch_chat(user_id, session_id, agent_id)

    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello"

@pytest.mark.asyncio
async def test_fetch_all_chats(storage):
    user_id = "user1"
    session_id = "session1"
    agent1_id = "agent1"
    agent2_id = "agent2"
    message1 = ConversationMessage(role="user", content="Hello Agent 1")
    message2 = ConversationMessage(role="user", content="Hello Agent 2")

    await storage.save_chat_message(user_id, session_id, agent1_id, message1)
    await storage.save_chat_message(user_id, session_id, agent2_id, message2)

    result = await storage.fetch_all_chats(user_id, session_id)

    assert len(result) == 2
    assert result[0].content == "Hello Agent 1"
    assert result[1].content == "Hello Agent 2"

@pytest.mark.asyncio
async def test_fetch_all_chats(storage):
    user_id = "user1"
    session_id = "session1"
    agent1_id = "agent1"
    agent2_id = "agent2"
    message1 = ConversationMessage(role="user", content=[{'text':"Hello"}])
    message2 = ConversationMessage(role="assistant", content=[{'text':"Hello Agent 2"}])

    await storage.save_chat_message(user_id, session_id, agent1_id, message1)
    await storage.save_chat_message(user_id, session_id, agent2_id, message2)

    result = await storage.fetch_all_chats(user_id, session_id)
    assert len(result) == 2
    assert result[0].content == [{'text':"Hello"}]
    assert result[1].content == [{'text':"[agent2] Hello Agent 2"}]

@pytest.mark.asyncio
async def test_trim_conversation(storage):
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"

    for i in range(5):
        await storage.save_chat_message(user_id, session_id, agent_id, ConversationMessage(role="user", content=f"Message {i}"))
        await storage.save_chat_message(user_id, session_id, agent_id, ConversationMessage(role="assistant", content=f"Message {i}"))

    result = await storage.fetch_chat(user_id, session_id, agent_id, max_history_size=3)

    assert len(result) == 2
    assert result[0].content == "Message 4"
    assert result[0].role == "user"
    assert result[1].content == "Message 4"
    assert result[1].role == "assistant"

def test_generate_key():
    key = InMemoryChatStorage._generate_key("user1", "session1", "agent1")
    assert key == "user1#session1#agent1"

def test_remove_timestamps():
    timestamped_messages = [
        TimestampedMessage(role="user", content="Hello", timestamp=1234567890),
        TimestampedMessage(role="agent", content="Hi", timestamp=1234567891)
    ]
    result = InMemoryChatStorage._remove_timestamps(timestamped_messages)

    assert len(result) == 2
    assert isinstance(result[0], ConversationMessage)
    assert result[0].role == "user"
    assert result[0].content == "Hello"
    assert not hasattr(result[0], 'timestamp')


@pytest.mark.asyncio
async def test_save_chat_messages(storage):
    """
    Testing saving multiple ConversationMessage at once
    """
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    messages = [ConversationMessage(role="user", content="Hello"), ConversationMessage(role="assistant", content='Hello from assistant')]

    result = await storage.save_chat_messages(user_id, session_id, agent_id, messages)

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "Hello"
    assert result[1].role == "assistant"
    assert result[1].content == "Hello from assistant"

@pytest.mark.asyncio
async def test_save_chat_messages_timestamp(storage):
    """
    Testing saving multiple TimestampedMessage at once
    """
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    messages = [TimestampedMessage(role="user", content="Hello"), TimestampedMessage(role="assistant", content='Hello from assistant')]

    result = await storage.save_chat_messages(user_id, session_id, agent_id, messages)

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "Hello"
    assert result[1].role == "assistant"
    assert result[1].content == "Hello from assistant"

@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_supplied_timestamps_preserve_cross_agent_order(storage, batch):
    for text, timestamp in [("newer", 2000), ("older", 1000)]:
        message = TimestampedMessage(role="user", content=[{"text": text}], timestamp=timestamp)
        if batch:
            await storage.save_chat_messages("u", "s", text, [message])
        else:
            await storage.save_chat_message("u", "s", text, message)
        message.timestamp = -1  # Saving snapshots the supplied timestamp value.

    stored_timestamps = [m.timestamp for history in storage.conversations.values() for m in history]
    assert sorted(stored_timestamps) == [1000, 2000]
    history = await storage.fetch_all_chats("u", "s")
    assert [m.content[0]["text"] for m in history] == ["older", "newer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("plain_first", [False, True])
async def test_batch_preserves_each_supplied_timestamp(storage, plain_first):
    messages = [
        TimestampedMessage(role="user", content=[{"text": "newer"}], timestamp=2000),
        TimestampedMessage(role="user", content=[{"text": "older"}], timestamp=1000),
    ]
    generated = ConversationMessage(role="user", content=[{"text": "generated"}])
    messages.insert(0 if plain_first else len(messages), generated)
    await storage.save_chat_messages("u", "s", "a", messages)

    stored = next(iter(storage.conversations.values()))
    for original, persisted in zip(messages, stored, strict=True):
        if isinstance(original, TimestampedMessage):
            assert persisted.timestamp == original.timestamp
        else:
            assert persisted.timestamp > 2000
    history = await storage.fetch_all_chats("u", "s")
    assert [m.content[0]["text"] for m in history] == ["older", "newer", "generated"]
