import pytest
from typing import Union
from agent_squad.types import ConversationMessage, TimestampedMessage
from agent_squad.storage import ChatStorage

class MockChatStorage(ChatStorage):
    async def save_chat_message(self, user_id: str, session_id: str, agent_id: str, new_message: Union[ConversationMessage, TimestampedMessage], max_history_size: int = None) -> bool:
        assert isinstance(new_message, ConversationMessage) or isinstance(new_message, TimestampedMessage)
        return True

    async def save_chat_messages(self, user_id: str, session_id: str, agent_id: str, new_messages: Union[list[ConversationMessage], list[TimestampedMessage]], max_history_size: int = None) -> bool:
        return True

    async def fetch_chat(self, user_id: str, session_id: str, agent_id: str, max_history_size: int = None) -> list[ConversationMessage]:
        return []

    async def fetch_all_chats(self, user_id: str, session_id: str) -> list[ConversationMessage]:
        return []

@pytest.fixture
def chat_storage():
    return MockChatStorage()

def test_is_same_role_as_last_message(chat_storage):
    conversation = [
        ConversationMessage(role="user", content="Hello"),
        ConversationMessage(role="assistant", content="Hi there"),
    ]

    # Test consecutive message
    new_message = ConversationMessage(role="assistant", content="How can I help you?")
    assert chat_storage.is_same_role_as_last_message(conversation, new_message) == True

    # Test non-consecutive message
    new_message = ConversationMessage(role="user", content="I have a question")
    assert chat_storage.is_same_role_as_last_message(conversation, new_message) == False

    # Test empty conversation
    assert chat_storage.is_same_role_as_last_message([], new_message) == False

def test_trim_conversation(chat_storage):
    conversation = [
        ConversationMessage(role="user", content="Message 1"),
        ConversationMessage(role="assistant", content="Response 1"),
        ConversationMessage(role="user", content="Message 2"),
        ConversationMessage(role="assistant", content="Response 2"),
        ConversationMessage(role="user", content="Message 3"),
        ConversationMessage(role="assistant", content="Response 3"),
    ]

    # Test with even max_history_size
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=4)
    assert len(trimmed) == 4
    assert trimmed[0].content == "Message 2"
    assert trimmed[-1].content == "Response 3"

    # Test with odd max_history_size (should adjust to even)
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=5)
    assert len(trimmed) == 4
    assert trimmed[0].content == "Message 2"
    assert trimmed[-1].content == "Response 3"

    # Test with max_history_size larger than conversation
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=10)
    assert len(trimmed) == 6
    assert trimmed == conversation

    # Test with None max_history_size
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=None)
    assert trimmed == conversation


def test_trim_conversation_zero_and_one_keep_nothing(chat_storage):
    """A size of 0 (or 1 rounded down to 0) must not keep the whole history."""
    conversation = [
        ConversationMessage(role="user", content="Message 1"),
        ConversationMessage(role="assistant", content="Response 1"),
        ConversationMessage(role="user", content="Message 2"),
        ConversationMessage(role="assistant", content="Response 2"),
    ]

    # conversation[-0:] is conversation[:], so without an explicit guard
    # these return the full conversation instead of an empty list.
    assert chat_storage.trim_conversation(conversation, max_history_size=0) == []
    assert chat_storage.trim_conversation(conversation, max_history_size=1) == []

    # Sizes >= 2 keep the existing even-pair behaviour.
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=2)
    assert len(trimmed) == 2
    assert trimmed[-1].content == "Response 2"

@pytest.mark.asyncio
async def test_save_chat_message(chat_storage):
    result = await chat_storage.save_chat_message(
        user_id="user1",
        session_id="session1",
        agent_id="agent1",
        new_message=ConversationMessage(role="user", content="Test message"),
        max_history_size=10
    )
    assert result == True

@pytest.mark.asyncio
async def test_save_chat_messages(chat_storage):
    result = await chat_storage.save_chat_messages(
        user_id="user1",
        session_id="session1",
        agent_id="agent1",
        new_messages=[
            ConversationMessage(role="user", content="Test message"),
            ConversationMessage(role="assitant", content="Test message from assistant")],
        max_history_size=10
    )
    assert result == True

    result = await chat_storage.save_chat_messages(
        user_id="user1",
        session_id="session1",
        agent_id="agent1",
        new_messages=[
            TimestampedMessage(role="user", content="Test message"),
            TimestampedMessage(role="assitant", content="Test message from assistant")],
        max_history_size=10
    )
    assert result == True

@pytest.mark.asyncio
async def test_fetch_chat(chat_storage):
    chat = await chat_storage.fetch_chat(
        user_id="user1",
        session_id="session1",
        agent_id="agent1",
        max_history_size=10
    )
    assert isinstance(chat, list)

@pytest.mark.asyncio
async def test_fetch_all_chats(chat_storage):
    chats = await chat_storage.fetch_all_chats(
        user_id="user1",
        session_id="session1"
    )
    assert isinstance(chats, list)