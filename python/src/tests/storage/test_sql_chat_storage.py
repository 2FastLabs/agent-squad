import os
import pytest
import tempfile
import pytest_asyncio
from agent_squad.storage import SqlChatStorage
from agent_squad.types import ConversationMessage, ParticipantRole, TimestampedMessage

# Configure pytest-asyncio to use asyncio mode
pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture(scope="function")
async def sql_storage():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        db_path = temp_db.name

    storage = SqlChatStorage(f"file:{db_path}")
    await storage.initialize()
    yield storage

    await storage.close()
    if os.path.exists(db_path):
        os.unlink(db_path)

@pytest.mark.asyncio
async def test_save_and_fetch_message(sql_storage: SqlChatStorage):
    """Test saving and fetching a single message."""
    message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"text": "Hello, world!"}]
    )

    # Save message
    saved_messages = await sql_storage.save_chat_message(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
        new_message=message
    )
    assert len(saved_messages) == 1
    assert saved_messages[0].role == message.role
    assert saved_messages[0].content == message.content

    # Fetch and verify
    messages = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent"
    )
    assert len(messages) == 1
    assert messages[0].role == message.role
    assert messages[0].content == message.content

@pytest.mark.asyncio
async def test_save_consecutive_same_role_messages(sql_storage: SqlChatStorage):
    """Test that consecutive messages with the same role are not saved."""
    message1 = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"text": "First message"}]
    )
    message2 = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"text": "Second message"}]
    )

    # Save first message
    saved_messages1 = await sql_storage.save_chat_message(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
        new_message=message1
    )
    assert len(saved_messages1) == 1
    assert saved_messages1[0].content == message1.content

    # Try to save second message with same role
    saved_messages2 = await sql_storage.save_chat_message(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
        new_message=message2
    )
    assert len(saved_messages2) == 1
    assert saved_messages2[0].content == message1.content

    # Verify only first message was saved
    messages = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent"
    )
    assert len(messages) == 1
    assert messages[0].content == message1.content

@pytest.mark.asyncio
async def test_max_history_size(sql_storage: SqlChatStorage):
    """Test that max_history_size properly limits the number of messages."""
    messages = [
        ConversationMessage(
            role=ParticipantRole.USER.value if i % 2 == 0 else ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Message {i}"}]
        ) for i in range(5)
    ]

    # Save all messages
    for msg in messages:
        await sql_storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id="test_agent",
            new_message=msg,
            max_history_size=3
        )

    # Verify only last 3 messages are kept
    fetched = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent"
    )
    assert len(fetched) == 3
    assert [m.content[0]["text"] for m in fetched] == ["Message 2", "Message 3", "Message 4"]

@pytest.mark.asyncio
async def test_save_multiple_messages(sql_storage: SqlChatStorage):
    """Test saving multiple messages in a batch."""
    messages = [
        ConversationMessage(
            role=ParticipantRole.USER.value if i % 2 == 0 else ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Message {i}"}]
        ) for i in range(3)
    ]

    # Save messages in batch
    saved_messages = await sql_storage.save_chat_messages(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
        new_messages=messages
    )
    assert len(saved_messages) == 3
    for i, msg in enumerate(saved_messages):
        assert msg.content[0]["text"] == f"Message {i}"

    # Verify all messages were saved
    fetched = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent"
    )
    assert len(fetched) == 3
    for i, msg in enumerate(fetched):
        assert msg.content[0]["text"] == f"Message {i}"

@pytest.mark.asyncio
async def test_fetch_all_chats(sql_storage: SqlChatStorage):
    """Test fetching all chats across different agents."""
    # Save messages for different agents
    agents = ["agent1", "agent2"]
    for agent_id in agents:
        message = ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Message from {agent_id}"}]
        )
        await sql_storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id=agent_id,
            new_message=message
        )

    # Fetch all chats
    all_messages = await sql_storage.fetch_all_chats(
        user_id="test_user",
        session_id="test_session"
    )

    assert len(all_messages) == 2
    agent_messages = [msg.content[0]["text"] for msg in all_messages]
    assert "[agent1] Message from agent1" in agent_messages
    assert "[agent2] Message from agent2" in agent_messages

@pytest.mark.asyncio
async def test_multiple_users_and_sessions(sql_storage: SqlChatStorage):
    """Test handling multiple users and sessions independently."""
    test_cases = [
        ("user1", "session1", "Hello from user1 session1"),
        ("user1", "session2", "Hello from user1 session2"),
        ("user2", "session1", "Hello from user2 session1")
    ]

    # Save messages for different user/session combinations
    for user_id, session_id, text in test_cases:
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": text}]
        )
        await sql_storage.save_chat_message(
            user_id=user_id,
            session_id=session_id,
            agent_id="test_agent",
            new_message=message
        )

    # Verify each combination independently
    for user_id, session_id, text in test_cases:
        messages = await sql_storage.fetch_chat(
            user_id=user_id,
            session_id=session_id,
            agent_id="test_agent"
        )
        assert len(messages) == 1
        assert messages[0].content[0]["text"] == text

@pytest.mark.asyncio
async def test_empty_fetch(sql_storage: SqlChatStorage):
    """Test fetching from empty storage."""
    messages = await sql_storage.fetch_chat(
        user_id="nonexistent",
        session_id="nonexistent",
        agent_id="nonexistent"
    )
    assert len(messages) == 0

    all_messages = await sql_storage.fetch_all_chats(
        user_id="nonexistent",
        session_id="nonexistent"
    )
    assert len(all_messages) == 0

@pytest.mark.asyncio
async def test_save_empty_batch(sql_storage: SqlChatStorage):
    """Test saving an empty batch of messages."""
    saved_messages = await sql_storage.save_chat_messages(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
        new_messages=[]
    )
    assert len(saved_messages) == 0

@pytest.mark.asyncio
async def test_message_ordering(sql_storage: SqlChatStorage):
    """Test that messages are returned in correct chronological order."""
    messages = [
        ConversationMessage(
            role=ParticipantRole.USER.value if i % 2 == 0 else ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Message {i}"}]
        ) for i in range(5)
    ]

    # Save messages one by one
    for msg in messages:
        await sql_storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id="test_agent",
            new_message=msg
        )

    # Verify order with and without max_history_size
    full_history = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent"
    )
    assert [m.content[0]["text"] for m in full_history] == [f"Message {i}" for i in range(5)]

    limited_history = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
        max_history_size=3
    )
    assert [m.content[0]["text"] for m in limited_history] == [f"Message {i}" for i in range(2, 5)]

@pytest.mark.asyncio
async def test_invalid_database_url():
    """Test handling of invalid database URL."""
    with pytest.raises(Exception):
        storage = SqlChatStorage("invalid://url")
        await storage.initialize()

@pytest.mark.asyncio
async def test_concurrent_access(sql_storage: SqlChatStorage):
    """Test concurrent access to the database."""
    import asyncio

    async def save_message(i: int):
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"Concurrent message {i}"}]
        )
        await sql_storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id=f"agent{i}",
            new_message=message
        )

    # Run multiple saves concurrently
    await asyncio.gather(*[save_message(i) for i in range(5)])

    # Verify all messages were saved
    all_messages = await sql_storage.fetch_all_chats(
        user_id="test_user",
        session_id="test_session"
    )
    assert len(all_messages) == 5
    messages = [msg.content[0]["text"] for msg in all_messages]
    for i in range(5):
        assert f"Concurrent message {i}" in messages

@pytest.mark.asyncio
async def test_transaction_rollback(sql_storage: SqlChatStorage):
    """Test transaction rollback on error."""
    messages = [
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": "Valid message"}]
        ),
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=None  # This will cause a JSON serialization error
        )
    ]

    # Attempt to save messages that will cause an error
    with pytest.raises(Exception):
        await sql_storage.save_chat_messages(
            user_id="test_user",
            session_id="test_session",
            agent_id="test_agent",
            new_messages=messages
        )

    # Verify no messages were saved due to transaction rollback
    saved_messages = await sql_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent"
    )
    assert len(saved_messages) == 0

@pytest.mark.asyncio
async def test_database_connection_handling():
    """Test proper handling of database connections."""
    with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
        storage = SqlChatStorage(f"file:{temp_db.name}")
        await storage.initialize()

        # Test basic operation
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": "Test message"}]
        )
        await storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id="test_agent",
            new_message=message
        )

        # Close connection
        await storage.close()

        # Verify operations fail after close
        with pytest.raises(Exception):
            await storage.save_chat_message(
                user_id="test_user",
                session_id="test_session",
                agent_id="test_agent",
                new_message=message
            )

@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_supplied_timestamps_preserve_cross_agent_order(sql_storage, batch):
    for text, timestamp in [("newer", 2000), ("older", 1000)]:
        message = TimestampedMessage(role="user", content=[{"text": text}], timestamp=timestamp)
        if batch:
            await sql_storage.save_chat_messages("u", "s", text, [message])
        else:
            await sql_storage.save_chat_message("u", "s", text, message)
        message.timestamp = -1  # Saving snapshots the supplied timestamp value.

    rows = await sql_storage.client.execute("SELECT timestamp FROM conversations ORDER BY timestamp")
    assert [row["timestamp"] for row in rows] == [1000, 2000]
    history = await sql_storage.fetch_all_chats("u", "s")
    assert [m.content[0]["text"] for m in history] == ["older", "newer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("plain_first", [False, True])
async def test_batch_preserves_each_supplied_timestamp(sql_storage, plain_first):
    messages = [
        TimestampedMessage(role="user", content=[{"text": "newer"}], timestamp=2000),
        TimestampedMessage(role="user", content=[{"text": "older"}], timestamp=1000),
    ]
    generated = ConversationMessage(role="user", content=[{"text": "generated"}])
    messages.insert(0 if plain_first else len(messages), generated)
    await sql_storage.save_chat_messages("u", "s", "a", messages)

    rows = await sql_storage.client.execute("SELECT timestamp FROM conversations ORDER BY message_index")
    for original, row in zip(messages, rows, strict=True):
        if isinstance(original, TimestampedMessage):
            assert row["timestamp"] == original.timestamp
        else:
            assert row["timestamp"] > 2000
    history = await sql_storage.fetch_all_chats("u", "s")
    assert [m.content[0]["text"] for m in history] == ["older", "newer", "generated"]
